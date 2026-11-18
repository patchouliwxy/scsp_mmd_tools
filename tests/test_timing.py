"""Different source durations must preserve rhythm, not truncate or time-stretch."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import motion_pipeline
from scsp_motion.backend import write_json
from scsp_motion.timing import frame_count,plan_song_timeline,hold_pose_tail
from scsp_motion.config import load_rules
from scsp_motion.expressions import mouth_for


def root(duration):
    curve=dict(m_Curve=[dict(time=0.,value=0.,inSlope=1.,outSlope=1.),
                       dict(time=duration,value=duration,inSlope=1.,outSlope=1.)])
    return dict(pos_x=curve,pos_z=curve)


class TimingTests(unittest.TestCase):
    def test_unity_float_noise_does_not_add_a_whole_frame(self):
        self.assertEqual(frame_count(147.43333435058594),4424)
        self.assertEqual(frame_count(147.43333),4424)

    def test_true_half_frame_end_is_preserved(self):
        self.assertEqual(frame_count(136.4166717529297),4094)
        self.assertEqual(frame_count(123.08334350585938),3694)

    def test_superd_plan_keeps_longest_body_and_records_short_tails(self):
        durations=[136.4166717529297,123.08334350585938,136.4166717529297,
                   136.4166717529297,136.4166717529297,123.08334350585938]
        slots=[dict(slot=i+1,duration=d) for i,d in enumerate(durations)]
        plan=plan_song_timeline(slots,{s['slot']:root(s['duration']) for s in slots})
        self.assertEqual(plan['frame_count'],4094)
        self.assertAlmostEqual(plan['slots']['2']['body_hold_tail_seconds'],13.333328247070312)
        self.assertEqual(plan['slots']['6']['source_frame_count'],3694)
        self.assertEqual(plan['slots']['1']['body_hold_tail_seconds'],0)

    def test_hold_changes_only_tail_samples(self):
        p=np.arange(18,dtype=float).reshape(3,2,3)
        q=np.tile([0.,0,0,1.],(3,2,1));q[1,0]=[0,0,1.,0]
        a,b=hold_pose_tail(p,q,7)
        np.testing.assert_array_equal(a[:3],p);np.testing.assert_array_equal(b[:3],q)
        np.testing.assert_array_equal(a[3:],np.repeat(p[-1:],4,axis=0))
        np.testing.assert_array_equal(b[3:],np.repeat(q[-1:],4,axis=0))

    def test_pose_is_never_truncated(self):
        with self.assertRaises(ValueError):hold_pose_tail(np.zeros((3,1,3)),np.ones((3,1,4)),2)

    def test_mismatched_root_for_same_person_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'本人的根位移'):
            plan_song_timeline([dict(slot=1,duration=2.)],{1:root(1.)})

    def test_strict_policy_can_reject_unequal_clips(self):
        with self.assertRaisesRegex(ValueError,'strict'):
            plan_song_timeline([dict(slot=1,duration=2.),dict(slot=2,duration=1.)],{1:root(2.),2:root(1.)},policy='strict')

    def test_luca_ruka_alias_is_explicit_and_ambiguous_files_are_rejected(self):
        info=dict(task_dir='.',mouths=['mouth/song_ruka_mouth.json'])
        self.assertEqual(mouth_for(dict(slot=6,character_id=26),info,load_rules()).name,'song_ruka_mouth.json')
        info['mouths'].append('mouth/song_luca_mouth.json')
        with self.assertRaises(ValueError):mouth_for(dict(slot=6,character_id=26),info,load_rules())

    def test_pipeline_aligns_body_root_and_expression_without_changing_sample_times(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);task=folder/'task';task.mkdir()
            mouths=[];slots=[]
            for number,duration,chara,alias in [(1,2.,14,'amana'),(2,1.,12,'rinze')]:
                body=task/f'body{number}.anim';body.write_text('fixture')
                rp=task/f'root{number}.bin';rp.write_bytes(b'fixture')
                exp=task/f'exp{number}.json'
                # This blink/close happens AFTER the second body's last frame.
                write_json(exp,dict(events=[dict(eventName='CloseEye',startFrame=90,arg1='0',arg2='100')]))
                mouth=task/f'song_{alias}_mouth.json'
                write_json(mouth,dict(frameRate=60,offsetFrame=0,datas=[dict(a=i/120) for i in range(121)]))
                mouths.append(str(mouth))
                slots.append(dict(slot=number,character_id=chara,duration=duration,body=str(body),root=str(rp),expression=str(exp)))
            info=dict(schema=1,task_dir=str(task),song='fixture',slots=slots,mouths=mouths,camera=None,warnings=[])
            rules=folder/'rules.json';write_json(rules,dict(camera_mode='off'))
            captured={}
            class Backend:
                identity={'type':'fixture'}
                model={}
                def calibrate(self,work):return dict(root_xz_to_world=[[1,0,0],[0,0,1]])
                def solve(self,request,work):
                    n=request['frameCount'];p=np.zeros((n,1,3));p[:,0,0]=np.arange(n)/30
                    return p,np.tile([0.,0,0,1.],(n,1,1)),{}
            def request(path,model):
                d=2. if path.stem=='body1' else 1.
                return dict(name=path.stem,duration=d,frameCount=frame_count(d),channels=[],omittedChannels=[])
            def convert(model,source,cal,p,q,root_motion,rules,mappings):
                number=len(captured)+1;captured[number]=(p.copy(),root_motion.copy())
                n=len(p);head=next(i for i,b in enumerate(model.bones) if b.name=='頭')
                values=np.c_[p[:,0],q[:,0]]
                return {'センター':(np.arange(n),values)},[],{'mapped_bones':['頭']},{head:np.zeros((n,3))}
            with patch('motion_pipeline.inspect_task',return_value=info), \
                 patch('motion_pipeline.root_curves',side_effect=lambda p:root(2. if 'root1' in p else 1.)), \
                 patch('motion_pipeline.humanoid_input',side_effect=request), \
                 patch('motion_pipeline.convert_pose',side_effect=convert):
                result=motion_pipeline.build_task(task,folder/'out',backend=Backend(),rules_path=rules)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['timeline']['frame_count'],61)
            for c in result['characters']:
                self.assertEqual(c['validation']['last_frame'],60)
            # Body 2 still reaches its original half-second sample at frame 15,
            # and frame 30..60 holds its t=1 pose and root, rather than stretching.
            p,trajectory=captured[2]
            self.assertAlmostEqual(p[15,0,0],.5)
            np.testing.assert_array_equal(p[30:,0,0],np.ones(31))
            np.testing.assert_array_equal(trajectory[30:],np.repeat(trajectory[30:31],31,axis=0))
            self.assertIn('ウィンク２',result['characters'][1]['validation']['names']['morphs'])
            self.assertEqual(result['characters'][1]['timing']['source_frame_count'],31)


if __name__=='__main__':unittest.main()
