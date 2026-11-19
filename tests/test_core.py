import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from unittest.mock import patch
from scsp_motion import vmd,pmx_io
from scsp_motion.curves import sample,animator_curves
from scsp_motion.math3d import rotate,mul,inv,between,world_pose,euler_unity
from scsp_motion.config import load_rules,ROOT
from scsp_motion.retarget import convert_pose
from scsp_motion.expressions import bake_morphs,bake_gaze,mouth_for
from scsp_motion.camera import bake_camera
from scsp_motion.task import safe_file


class CoreTests(unittest.TestCase):
    def test_hermite_uses_slopes(self):
        c=[dict(time=0,value=0,inSlope=0,outSlope=2),dict(time=1,value=0,inSlope=0,outSlope=0)]
        self.assertAlmostEqual(sample(c,[.5])[0],.25)

    def test_step_exact_endpoint(self):
        c=[dict(time=0,value=2,outSlope=float('inf')),dict(time=1,value=5,inSlope=float('inf'))]
        np.testing.assert_equal(sample(c,[-1,.5,1,2]),[2,2,5,5])

    def test_weighted_bezier_inverts_time(self):
        c=[dict(time=0,value=0,outSlope=2,weightedMode=2,outWeight=.1),
           dict(time=1,value=1,inSlope=0,weightedMode=1,inWeight=.2)]
        u=.4
        t=3*(1-u)**2*u*.1+3*(1-u)*u*u*.8+u**3
        y=3*(1-u)**2*u*.2+3*(1-u)*u*u+u**3
        self.assertAlmostEqual(sample(c,[t])[0],y,places=8)

    def test_curve_rejects_duplicate_times(self):
        with self.assertRaises(ValueError):sample([dict(time=0,value=1),dict(time=0,value=2)],[0])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):sample([dict(time=0,value=float('nan'))],[0])

    def test_quaternion_noncommuting(self):
        x=np.array([np.sin(.4),0,0,np.cos(.4)]);y=np.array([0,np.sin(.7),0,np.cos(.7)])
        p=np.array([1.,2,3])
        np.testing.assert_allclose(rotate(mul(x,y),p),rotate(x,rotate(y,p)),atol=1e-12)
        np.testing.assert_allclose(rotate(inv(x),rotate(x,p)),p,atol=1e-12)

    def test_opposite_vector_alignment(self):
        q=between([1.,0,0],[-1.,0,0]);np.testing.assert_allclose(rotate(q,[1.,0,0]),[-1,0,0],atol=1e-10)

    def test_hierarchy_retains_fbx_scale(self):
        nodes=[dict(parent=-1,position=[1,0,0],rotation=[0,0,0,1],scale=[100]*3),
               dict(parent=0,position=[0,.01,0],rotation=[0,0,0,1],scale=[1]*3)]
        p,q=world_pose(nodes);np.testing.assert_allclose(p[0,1],[1,1,0])

    def test_vmd_roundtrip_all_sections_and_strict_names(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'motion.vmd';q=np.array([[0,0,0,0,0,0,1],[1,2,3,0,0,0,1.]])
            r=vmd.write(p,'test',{'センター':([0,10],q)},{'あ':([0,10],[0.,1.])},
                        [dict(frame=0,distance=-30,position=[0,10,0],rotation=[0,0,0],fov=35)],[(0,{'左足ＩＫ':False})])
            self.assertEqual(r['counts'],dict(bones=2,morphs=2,cameras=1,lights=0,shadows=0,ik=1))
            self.assertEqual(r['names']['bones'],['センター'])
            with self.assertRaises(FileExistsError):vmd.write(p,'test')
            with self.assertRaises(ValueError):vmd.name_bytes('あ'*8,15)
            data=p.read_bytes();p.write_bytes(data[:-1])
            with self.assertRaises(ValueError):vmd.inspect(p)

    def test_camera_only_zero_bone_file(self):
        with tempfile.TemporaryDirectory() as t:
            r=vmd.write(Path(t)/'cam.vmd','カメラ・照明',cameras=[dict(frame=0,distance=0,position=[1,2,3],rotation=[0,0,0],fov=45)])
            self.assertEqual(r['counts']['cameras'],1)

    def test_duplicate_vmd_frames_rejected(self):
        with self.assertRaises(ValueError):vmd.validate_frames([0,0])

    def test_constant_reduction_preserves_edges(self):
        np.testing.assert_equal(vmd.stable_indices(np.array([[0],[0],[0],[1],[1],[2]])),[0,2,3,4,5])

    def test_input_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):safe_file(t,'../LICENSE')

    def test_rules_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json';p.write_text('{"fpps":60}')
            with self.assertRaises(ValueError):load_rules(p)

    def test_interrupted_eye_transition_and_mouth_timing(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);e=t/'e.json';m=t/'m.json'
            e.write_text(json.dumps(dict(events=[dict(eventName='CloseEye',startFrame=0,arg1='20',arg2='100'),
                      dict(eventName='OpenEye',startFrame=10,arg1='10',arg2='0'),
                      dict(eventName='CloseEye',startFrame=100001,arg1='0',arg2='100')])) )
            m.write_text(json.dumps(dict(frameRate=60,offsetFrame=8,datas=[dict(a=1),dict(a=0),dict(a=.5)])))
            model=pmx_io.Model();names=json.loads((ROOT/'Resources/unity_morph_names.json').read_text(encoding='utf-8'))
            for n in names['source_names'].values():
                morph=pmx_io.VertexMorph();morph.name=n;model.morphs.append(morph)
            tracks,report,gaze=bake_morphs(e,m,16,load_rules(),model,names)
            fs,vs=tracks['ウィンク２'];self.assertAlmostEqual(np.interp(5,fs,vs),.5)
            self.assertAlmostEqual(np.interp(8,fs,vs),.2)
            fs,vs=tracks['あ'];self.assertEqual(vs[0],1);self.assertEqual(np.interp(1,fs,vs),.5)
            self.assertEqual(report['source_offset_frame'],8);self.assertEqual(report['applied_offset_frames'],0)
            self.assertEqual(report['events_beyond_clip'],1)

    def test_mouth_routing_uses_character_not_slot(self):
        info=dict(task_dir='.',mouths=['mouth/song_amana_mouth.json','mouth/song_rinze_mouth.json'])
        p=mouth_for(dict(slot=1,character_id=14),info,load_rules());self.assertIn('amana',str(p))

    def test_camera_relative_anchor_and_origin(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera.json'
            p.write_text(json.dumps(dict(cameraWorkData=[dict(startFrame=0,frameRate=60,moveID='a',useStartTargetData=True,
                 targetBasePoint=dict(x=10,y=0,z=2),targetNormalized=dict(x=0,y=1,z=0))],
                 customCameraMoveData=[dict(moveID='a',cameraPos=[dict(x=0,y=0,z=-2)],tiltPanRoll=[dict(x=10,y=20,z=0)],zoom=[45])])) )
            keys,pos,report=bake_camera(p,4,np.array([-10,0,-2]),np.array([[-1,0,0],[0,0,-1]]),load_rules())
            np.testing.assert_allclose(pos[0],[0,12.5,-25]);np.testing.assert_allclose(keys[0]['rotation'],np.deg2rad([-10,-20,0]))

    def test_gaze_mode_source_default_and_approximate_override(self):
        rules=load_rules()
        self.assertEqual(rules['gaze_mode'],'source')
        model=pmx_io.Model()
        for name,parent,pos in [('頭',-1,[0,10,0]),('左目',0,[0.3,10.5,-0.2]),('右目',0,[-0.3,10.5,-0.2])]:
            b=pmx_io.Bone();b.name=name;b.parent=parent;b.location=pos;model.bones.append(b)
        events=[dict(eventName='ChangeTarget',startFrame=0,arg1='CenterMiddle',arg2='0'),
                dict(eventName='ChangeTarget',startFrame=4,arg1='Camera',arg2='4')]
        world={0:np.tile([0,10.,0],(4,1)),1:np.tile([0.3,10.5,-0.2],(4,1)),2:np.tile([-0.3,10.5,-0.2],(4,1))}
        tracks,gaze_report=bake_gaze(events,model,world,{},{},None,rules)
        self.assertEqual(tracks,{})
        self.assertEqual(gaze_report['mode'],'source')
        self.assertFalse(gaze_report['applied_event_gaze'])
        self.assertEqual(gaze_report['events'],2)
        self.assertEqual(gaze_report['deferred_targets'],{'CenterMiddle':1,'Camera':1})
        approx_rules=dict(rules,gaze_mode='approximate')
        approx_tracks,approx_report=bake_gaze(events,model,world,{},{},None,approx_rules)
        self.assertIn('左目',approx_tracks);self.assertIn('右目',approx_tracks)
        self.assertEqual(approx_report['mode'],'approximate')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad_gaze.json';p.write_text('{"gaze_mode":"force"}')
            with self.assertRaises(ValueError):load_rules(p)

    def test_bake_gaze_head_relative_directions_behind_fade_and_conv_guard(self):
        from scsp_motion.math3d import rotate
        rules=load_rules();rules['gaze_mode']='approximate'
        model=pmx_io.Model()
        for name,parent,pos in [('頭',-1,[0,10,0]),('左目',0,[0.5,10.5,-0.2]),('右目',0,[-0.5,10.5,-0.2]),('左手首',-1,[0.5,10.5,-1.5])]:
            b=pmx_io.Bone();b.name=name;b.parent=parent;b.location=pos;model.bones.append(b)
        # Rotate head -90 deg around Y (now facing +X); cam at [-50, 10.5, 0] is directly behind head!
        q_head_90=np.array([0.,-np.sin(np.pi/4),0.,np.cos(np.pi/4)])
        bone_tracks={'頭':(np.array([0,1,2,3],dtype=np.uint32),np.tile(np.r_[0.,0.,0.,q_head_90],(4,1)))}
        world={
            0:np.tile([0,10.,0],(4,1)),
            1:np.tile([0.2,10.5,0.5],(4,1)),
            2:np.tile([0.2,10.5,-0.5],(4,1)),
            3:np.tile([1.5,10.5,0.0],(4,1)), # 1.3m in front along rotated head +X, closer than 1.5m min_conv_dist!
        }
        # Camera directly behind rotated head (-X) -> fade to 0!
        cam=np.tile([-50.,10.5,0.],(4,1))
        events=[
            dict(eventName='ChangeTarget',startFrame=0,arg1='CenterMiddle',arg2='0'),
            dict(eventName='ChangeTarget',startFrame=2,arg1='LeftMiddle',arg2='0'),
            dict(eventName='ChangeTarget',startFrame=4,arg1='BackMiddle',arg2='0'),
            dict(eventName='ChangeTarget',startFrame=6,arg1='Camera',arg2='0'),
        ]
        tracks,report=bake_gaze(events,model,world,bone_tracks,{},cam,rules)
        def eval_track(name,f):
            fs,vs=tracks[name]
            idx=int(np.searchsorted(fs,f,side='right')-1)
            return vs[max(0,idx),3:]
        # Frame 0: CenterMiddle stays neutral even when head is turned 90 deg
        np.testing.assert_allclose(rotate(eval_track('左目',0),[0.,0.,-1.]),[0.,0.,-1.],atol=1e-6)
        # Frame 1 (60fps frame 2): LeftMiddle looks toward +X (Head local left) within safe 5.5 deg anime bound!
        d_left=rotate(eval_track('左目',1),[0.,0.,-1.])
        self.assertGreater(d_left[0],0.08)
        self.assertLess(d_left[0],0.12)
        self.assertLess(abs(d_left[1]),1e-6)
        # Frame 2 (60fps frame 4): BackMiddle returns to neutral instead of flipping!
        np.testing.assert_allclose(rotate(eval_track('左目',2),[0.,0.,-1.]),[0.,0.,-1.],atol=1e-6)
        # Frame 3 (60fps frame 6): Behind-head Camera fades smoothly to neutral!
        np.testing.assert_allclose(rotate(eval_track('左目',3),[0.,0.,-1.]),[0.,0.,-1.],atol=1e-6)
        # Unresolved Target2 + arg2=-1 smooth transition + LeftHand convergence guard
        events2=[
            dict(eventName='ChangeTarget',startFrame=0,arg1='HalfLeftUp',arg2='0'),
            dict(eventName='ChangeTarget',startFrame=2,arg1='Target2',arg2='-1'),
        ]
        tracks2,report2=bake_gaze(events2,model,world,bone_tracks,{},cam,rules)
        self.assertEqual(report2['unknown_targets'],{'Target2':1})
        self.assertIn('左目',tracks2)
        # Verify close LeftHand doesn't cross eyes (左目 and 右目 agree on direction!)
        tracks_hand,_=bake_gaze([dict(eventName='ChangeTarget',startFrame=0,arg1='LeftHand',arg2='0')],model,world,bone_tracks,{},cam,rules)
        dl_h=rotate(tracks_hand['左目'][1][0,3:],[0.,0.,-1.])
        dr_h=rotate(tracks_hand['右目'][1][0,3:],[0.,0.,-1.])
        self.assertLess(np.linalg.norm(dl_h-dr_h),0.06)

    def test_camera_offset_rotate_transition_and_actor_anchor_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera.json'
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='BustUp',objectList=[1],
                         useStartTargetData=False,offsetRotate=dict(x=0,y=-90,z=0),moveTransition=0.0),
                    dict(startFrame=4,frameRate=60,moveID='b',shootingType='BustUp',objectList=[1],
                         useStartTargetData=True,targetBasePoint=dict(x=1,y=0,z=0),targetNormalized=dict(x=0,y=1,z=0),
                         offsetRotate=dict(x=0,y=0,z=0),moveTransition=0.2),
                ],
                customCameraMoveData=[
                    dict(moveID='a',cameraPos=[dict(x=0,y=0,z=-2),dict(x=0,y=0,z=-2)],
                         tiltPanRoll=[dict(x=0,y=0,z=0),dict(x=0,y=0,z=0)],zoom=[20,20]),
                    dict(moveID='b',cameraPos=[dict(x=2,y=0,z=0),dict(x=2,y=0,z=0)],
                         tiltPanRoll=[dict(x=0,y=270,z=0),dict(x=0,y=270,z=0)],zoom=[20,20]),
                ]
            )))
            # Actor 1 chest in MMD PMX units sits at (-12.5, 12.5, 0) plus BustUp offset (0, +1.25, -0.9375).
            actor_positions={1:{'上半身3':np.tile([-12.5,12.5,0.],(8,1))}}
            keys,pos,report=bake_camera(p,8,np.zeros(3),np.array([[-1,0,0],[0,0,-1]]),dict(load_rules(),camera_sparse_reduce=False),actor_positions=actor_positions)
            # Shot 'a' has pos=(0,0,-2) rotated by y=-90 -> (+2,0,0) in meters (+25 PMX) + BustUp anchor (-12.5,13.75,-0.9375).
            np.testing.assert_allclose(pos[0],[12.5,13.75,-0.9375],atol=1e-5)
            fwd0=rotate(euler_unity(-np.rad2deg(keys[0]['rotation'])),[0.,0.,1.])
            np.testing.assert_allclose(fwd0,[-1.,0.,0.],atol=1e-5)
            # Shot 'b' transitions smoothly from shot 'a' without a 360-degree yaw spin despite 270 vs -90 deg.
            self.assertEqual(report['applied_features']['offsetRotate'],1)
            self.assertEqual(report['applied_features']['objectListAnchorFallback'],1)
            self.assertEqual(report['applied_features']['moveTransition_approximate'],1)
            self.assertEqual(report['unsupported_features'],{})
            self.assertEqual([k['frame'] for k in keys],list(range(8)))
            self.assertEqual(keys[2]['interpolation'],bytes([20,107,20,107]*6))
            for k in keys:
                fwd=rotate(euler_unity(-np.rad2deg(k['rotation'])),[0.,0.,1.])
                self.assertGreater(fwd[0],-1.01)
                self.assertLess(abs(fwd[2]),1e-4)

    def test_camera_hard_cut_step_and_zoom_bezier_reduction(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'cam_zoom.json'
            n=60
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='Stage',useStartTargetData=False,moveTransition=0.0),
                    dict(startFrame=60,frameRate=60,moveID='b',shootingType='Stage',useStartTargetData=False,moveTransition=0.0),
                ],
                customCameraMoveData=[
                    dict(moveID='a',cameraPos=[dict(x=0,y=1,z=-5)]*n,
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*n,zoom=np.linspace(15.0,30.0,n).tolist()),
                    dict(moveID='b',cameraPos=[dict(x=5,y=2,z=-8)]*n,
                         tiltPanRoll=[dict(x=10,y=20,z=0)]*n,zoom=[35.0]*n),
                ]
            )))
            keys,pos,report=bake_camera(p,60,np.zeros(3),np.array([[-1,0,0],[0,0,-1]]),dict(load_rules(),camera_interpolation='legacy'))
            shot_a_keys=[k for k in keys if k['frame']<30]
            shot_b_keys=[k for k in keys if k['frame']>=30]
            # Linear zoom ramp over 30 VMD frames should reduce to endpoints, eliminating 1-deg integer stair-steps.
            self.assertEqual([k['frame'] for k in shot_a_keys],[0,29])
            self.assertEqual([k['frame'] for k in shot_b_keys],[30,59])
            self.assertEqual(shot_b_keys[0]['interpolation'],bytes([127,127,0,0]*6))

    def test_camera_target_height_scale_and_shooting_offsets(self):
        heights=json.loads((Path(__file__).resolve().parents[1]/'Resources'/'character_heights.json').read_text(encoding='utf-8'))
        hmap={int(r['id']):float(r['heightScale']) for r in heights['characters']}
        self.assertEqual(len(hmap),42)
        h25=hmap[25];h2=hmap[2]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera_height.json'
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='BustUp',objectList=[1],
                         useStartTargetData=True,targetBasePoint=dict(x=1,y=0,z=0),targetNormalized=dict(x=0,y=1,z=0),moveTransition=0.0),
                    dict(startFrame=4,frameRate=60,moveID='b',shootingType='UpperUp',objectList=[1,2],
                         useStartTargetData=True,targetBasePoint=dict(x=0,y=0,z=0),targetNormalized=dict(x=0,y=1,z=0),moveTransition=0.0),
                    dict(startFrame=8,frameRate=60,moveID='c',shootingType='RightHand',objectList=[1],
                         useStartTargetData=True,targetBasePoint=dict(x=0,y=0,z=0),targetNormalized=dict(x=0,y=1,z=0),moveTransition=0.0),
                ],
                customCameraMoveData=[
                    dict(moveID=mid,cameraPos=[dict(x=0,y=0,z=0)]*2,
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*2,zoom=[30,30])
                    for mid in ('a','b','c')
                ]
            )))
            basis=np.array([[-1,0,0],[0,0,-1]])
            keys,pos,report=bake_camera(p,6,np.zeros(3),basis,load_rules(),slot_characters={1:25,2:2})
            # Shot 'a': BustUp offset (0,+0.1,+0.075), single actor chara 25.
            np.testing.assert_allclose(pos[0],[-12.5,12.5*(h25+0.1),12.5*(-0.075)],atol=1e-5)
            # Shot 'b': UpperUp offset (0,0,+0.15), mean of chara 25 and chara 2.
            h_avg=0.5*(h25+h2)
            np.testing.assert_allclose(pos[2],[0.,12.5*h_avg,12.5*(-0.15)],atol=1e-5)
            # Shot 'c': RightHand has zero shooting offset, still scales targetNormalized by h25.
            np.testing.assert_allclose(pos[4],[0.,12.5*h25,0.],atol=1e-5)
            self.assertEqual(report['applied_features']['targetHeightScale'],3)
            self.assertEqual(report['applied_features']['shootingTypeOffset'],2)
            # Disabling height scale and clearing shooting offsets via custom rules restores unscaled base+norm.
            custom=Path(t)/'disable_offsets.json'
            custom.write_text(json.dumps(dict(camera_target_height_scale=False,camera_shooting_offsets={})))
            rules_off=load_rules(custom)
            self.assertEqual(rules_off['camera_shooting_offsets'],{})
            _,pos_off,report_off=bake_camera(p,6,np.zeros(3),basis,rules_off,slot_characters={1:25,2:2})
            np.testing.assert_allclose(pos_off[0],[-12.5,12.5,0.],atol=1e-5)
            np.testing.assert_allclose(pos_off[2],[0.,12.5,0.],atol=1e-5)
            self.assertNotIn('targetHeightScale',report_off['applied_features'])
            self.assertNotIn('shootingTypeOffset',report_off['applied_features'])

    def test_camera_dynamic_actor_anchor_tracks_moving_character(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera_dyn.json'
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='BustUp',objectList=[1],
                         useStartTargetData=False,moveTransition=0.0),
                ],
                customCameraMoveData=[
                    dict(moveID='a',cameraPos=[dict(x=0,y=0,z=-2)]*10,
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*10,zoom=[30.4]*10),
                ]
            )))
            head_traj=np.column_stack([np.linspace(0.,9.,10),np.full(10,15.),np.zeros(10)])
            actor_positions={1:{'頭':head_traj,'上半身3':np.zeros((10,3))}}
            basis=np.array([[-1,0,0],[0,0,-1]])
            keys,pos,report=bake_camera(p,10,np.zeros(3),basis,dict(load_rules(),camera_zoom_compensation=True),actor_positions=actor_positions)
            self.assertEqual(report['applied_features']['dynamicActorAnchor'],1)
            # Frame 9 anchor sits at x=9.0 + BustUp offset (0,+1.25,-0.9375); optical center tracks X and keeps compensated depth.
            np.testing.assert_allclose(pos[0,:2],[0.,16.25],atol=1e-4)
            np.testing.assert_allclose(pos[9,:2],[9.,16.25],atol=1e-4)
            depth9=(head_traj[9,2]-0.9375)-pos[9,2]
            expected_depth=25.*np.tan(np.deg2rad(30.4)/2)/np.tan(np.deg2rad(30)/2)
            self.assertAlmostEqual(depth9,expected_depth,places=3)
            # Turning off dynamic actor anchor keeps the shot anchored to its start frame.
            custom=Path(t)/'static_anchor.json'
            custom.write_text(json.dumps(dict(camera_dynamic_actor_anchor=False)))
            _,pos_static,report_static=bake_camera(p,10,np.zeros(3),basis,load_rules(custom),actor_positions=actor_positions)
            np.testing.assert_allclose(pos_static[:,0],np.zeros(10),atol=1e-4)
            self.assertNotIn('dynamicActorAnchor',report_static['applied_features'])

    def test_camera_fallback_respects_negative_movetrace_and_shooting_bones(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera_mellow_like.json'
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='UpperUp',objectList=[1],
                         moveTrace=-1.0,useStartTargetData=False,moveTransition=0.0),
                    dict(startFrame=10,frameRate=60,moveID='b',shootingType='FullBody',objectList=[1],
                         moveTrace=-1.0,useStartTargetData=True,targetBasePoint=dict(x=0,y=0,z=0),
                         targetNormalized=dict(x=0,y=1,z=0),moveTransition=0.0),
                ],
                customCameraMoveData=[
                    dict(moveID='a',cameraPos=[dict(x=0,y=0,z=-1)]*2,
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*2,zoom=[25.0]*2),
                    dict(moveID='b',cameraPos=[dict(x=0,y=0,z=0)]*2,
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*2,zoom=[25.0]*2),
                ]
            )))
            neck_traj=np.column_stack([np.linspace(1.,6.,10),np.full(10,16.),np.zeros(10)])
            actor_positions={1:{'首':neck_traj,'上半身3':np.full((10,3),[99.,12.,0.])}}
            basis=np.array([[-1,0,0],[0,0,-1]])
            _,pos,report=bake_camera(p,10,np.zeros(3),basis,load_rules(),actor_positions=actor_positions)
            # UpperUp uses '首' anchored statically at frame 0 (moveTrace=-1) plus UpperUp offset [0,0,+0.15] (-1.875 PMX Z).
            np.testing.assert_allclose(pos[:5,0],np.full(5,1.0),atol=1e-4)
            np.testing.assert_allclose(pos[:5,1],np.full(5,16.0),atol=1e-4)
            np.testing.assert_allclose(pos[:5,2],np.full(5,-12.5-1.875),atol=1e-4)
            self.assertNotIn('dynamicActorAnchor',report['applied_features'])
            # FullBody shooting offset Z=0.40 maps to -5.0 PMX under basis=[-1, -1].
            np.testing.assert_allclose(pos[5,:3],[0.,12.5,-5.0],atol=1e-4)

    def test_camera_noise_filter_attenuates_alternating_source_jitter(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera_noise.json'
            n=21
            zig=np.array([0. if i in (0,n-1) else (0.1 if i%2 else -0.1) for i in range(n)])
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='Stage',useStartTargetData=False,moveTransition=0.0),
                ],
                customCameraMoveData=[
                    dict(moveID='a',cameraPos=[dict(x=float(zig[i]),y=1.0,z=-5.0) for i in range(n)],
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*n,zoom=[30.0]*n),
                ]
            )))
            basis=np.array([[-1,0,0],[0,0,-1]])
            _,pos_raw,_=bake_camera(p,11,np.zeros(3),basis,load_rules())
            custom=Path(t)/'filt.json'
            custom.write_text(json.dumps(dict(camera_noise_filter=True)))
            _,pos_filt,rep_filt=bake_camera(p,11,np.zeros(3),basis,load_rules(custom))
            self.assertEqual(rep_filt['applied_features']['noiseFilter'],1)
            np.testing.assert_allclose(pos_filt[0],pos_raw[0],atol=1e-5)
            self.assertLess(np.max(np.abs(np.diff(pos_filt[:,0],n=2))),0.2*np.max(np.abs(np.diff(pos_raw[:,0],n=2))))

    def test_cli_build_without_pmx_invokes_generic_export(self):
        import motion_pipeline
        captured={}
        def fake_build(task_dir,output_root,**kwargs):
            captured['task_dir']=task_dir;captured['output_root']=output_root;captured.update(kwargs)
            return dict(status='completed',output_dir='out',validation=dict(binary_verified=True))
        with patch('motion_pipeline.build_task',side_effect=fake_build):
            rc=motion_pipeline.main(['build','task_dir','--output','out_dir','--backend','backend_dir'])
        self.assertEqual(rc,0)
        self.assertIsNone(captured['target_pmx'])

    def test_scenecreate_slot_remaps_permuted_body_root_exp_and_generic_mouth(self):
        from scsp_motion.task import inspect_task
        with tempfile.TemporaryDirectory() as t:
            td=Path(t)
            for sub in ('animations','data','mouth','root-data'):(td/sub).mkdir()
            (td/'job.json').write_text(json.dumps(dict(status='completed',dependency_complete=True,missing_count=0,
                mouth_files=['mouth/999_test_a_mouth.json','mouth/999_test_b_mouth.json','mouth/999_test_c_mouth.json'])))
            (td/'manifest.json').write_text(json.dumps(dict(song_selection=dict(code='999_test'))))
            (td/'data/999_test_unit.json').write_text(json.dumps(dict(unitIdolData=[dict(chara_id=16),dict(chara_id=15),dict(chara_id=14)])))
            (td/'data/999_test_scenecreate.json').write_text(json.dumps(dict(characterCreateData=[
                dict(animName='999_Test_01',lipSyncName='Test/999_test_c'),
                dict(animName='999_Test_03',lipSyncName='Test/999_test_b'),
                dict(animName='999_Test_02',lipSyncName='Test/999_test_a'),
            ])))
            sources=[];clips=[]
            for idx in (1,2,3):
                (td/f'animations/999_Test_{idx:02d}.anim').write_text('anim')
                (td/f'root-data/999_test_{idx:02d}_root.unity3d').write_text('root')
                (td/f'data/999_test_{idx:02d}_exp.json').write_text('{}')
                sources.append(dict(resource=f'999_test_{idx:02d}.anim',name=f'999_Test_{idx:02d}',humanoid_bindings=105,stop_time=10.0,sample_rate=60.0))
                clips.append(dict(file=f'animations/999_Test_{idx:02d}.anim'))
            for fn in ('999_test_a_mouth.json','999_test_b_mouth.json','999_test_c_mouth.json'):
                (td/f'mouth/{fn}').write_text('{}')
            (td/'animations/export-report.json').write_text(json.dumps(dict(sources=sources,clips=clips)))
            info=inspect_task(td)
            self.assertEqual(info['slot_binding_source'],'task_scenecreate')
            s2,s3=info['slots'][1],info['slots'][2]
            self.assertEqual((s2['slot'],s2['character_id'],s2['motion_slot']),(2,15,3))
            self.assertTrue(s2['body'].endswith('999_Test_03.anim'))
            self.assertTrue(s2['root'].endswith('999_test_03_root.unity3d'))
            self.assertTrue(s2['expression'].endswith('999_test_03_exp.json'))
            self.assertEqual(mouth_for(s2,info,load_rules()).name,'999_test_b_mouth.json')
            self.assertEqual((s3['slot'],s3['character_id'],s3['motion_slot']),(3,14,2))
            self.assertTrue(s3['body'].endswith('999_Test_02.anim'))

    def test_solo_song_without_slot_number_exp_and_root(self):
        from scsp_motion.task import inspect_task
        with tempfile.TemporaryDirectory() as t:
            td=Path(t)
            for sub in ('animations','data','mouth','root-data'):(td/sub).mkdir()
            (td/'job.json').write_text(json.dumps(dict(status='completed',dependency_complete=True,missing_count=0,
                mouth_files=['mouth/solo_chiyoko_mouth.json'])))
            (td/'manifest.json').write_text(json.dumps(dict(song_selection=dict(code='888_solo'))))
            (td/'data/888_solo_unit.json').write_text(json.dumps(dict(unitIdolData=[dict(chara_id=10)])))
            (td/'animations/888_Solo.anim').write_text('anim')
            (td/'root-data/888_solo_root.unity3d').write_text('root')
            (td/'data/888_solo_exp.json').write_text('{}')
            (td/'mouth/solo_chiyoko_mouth.json').write_text('{}')
            (td/'animations/export-report.json').write_text(json.dumps(dict(
                sources=[dict(resource='888_solo.anim',name='888_Solo',humanoid_bindings=105,stop_time=10.0,sample_rate=60.0)],
                clips=[dict(file='animations/888_Solo.anim')])))
            info=inspect_task(td)
            self.assertEqual(len(info['slots']),1)
            s1=info['slots'][0]
            self.assertTrue(s1['root'].endswith('888_solo_root.unity3d'))
            self.assertTrue(s1['expression'].endswith('888_solo_exp.json'))

    def test_camera_movespeed_power_exponent_and_partial_easing(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'camera_spd.json'
            # Cut has moveSpeed=0.5 and easing reaching 1.0 at time=0.5 (early plateau).
            # At t_raw=0.25, t_prog = 0.25 ** 0.5 = 0.5 -> reaches sample[-1] instead of stopping at 0.5*delta.
            p.write_text(json.dumps(dict(
                cameraWorkData=[
                    dict(startFrame=0,frameRate=60,moveID='a',shootingType='Stage',useStartTargetData=False,
                         moveSpeed=0.5,moveTransition=0.0,
                         easing=dict(m_Curve=[dict(time=0.0,value=0.0,inSlope=2.0,outSlope=2.0),
                                              dict(time=0.5,value=1.0,inSlope=2.0,outSlope=2.0)])),
                    dict(startFrame=32,frameRate=60,moveID='a',shootingType='Stage',useStartTargetData=False,
                         moveSpeed=1.0,moveTransition=0.0),
                ],
                customCameraMoveData=[
                    dict(moveID='a',
                         cameraPos=[dict(x=0,y=0,z=float(z)) for z in (0.,1.,2.,3.)],
                         tiltPanRoll=[dict(x=0,y=0,z=0)]*4,zoom=[30.0]*4),
                ]
            )))
            basis=np.array([[-1,0,0],[0,0,-1]])
            _,pos,_=bake_camera(p,17,np.zeros(3),basis,load_rules())
            # t_raw at frame 4/16 = 0.25 -> t_prog = 0.5 -> easing(0.5)=1.0 -> z = 3.0 * 12.5 = 37.5
            self.assertAlmostEqual(pos[4,2],37.5,places=4)
            self.assertAlmostEqual(pos[15,2],37.5,places=4)
            self.assertAlmostEqual(pos[16,2],0.0,places=4)

    def test_builtin_pose_backend_calibrates_and_matches_humanoid_kinematics(self):
        from scsp_motion.backend import BuiltinPoseBackend
        from scsp_motion.math3d import inv, mul
        bp=BuiltinPoseBackend()
        cal=bp.calibrate()
        self.assertGreater(cal['human_scale'],0.5)
        basis=np.asarray(cal['root_xz_to_world'])
        self.assertAlmostEqual(float(np.linalg.norm(basis[0])),1.0,places=4)
        self.assertAlmostEqual(float(np.linalg.norm(basis[1])),1.0,places=4)
        self.assertAlmostEqual(float(np.dot(basis[0],basis[1])),0.0,places=4)
        # Verify single-axis muscle deflection + RootT translation + omitted channels fallback.
        ch=['Spine Front-Back','Left Arm Down-Up','Left Forearm Stretch','RootT.x','RootT.y','RootT.z']
        samples=[[0.,0.,0.,0.,1.,0.],[1.,0.,0.,0.,1.,0.],[0.,1.,0.,1.,1.,0.]]
        p,q,meta=bp.solve(dict(name='t',fps=30,duration=2/30,frameCount=3,channels=ch,samples=samples))
        self.assertEqual(meta['frameCount'],3)
        self.assertEqual(meta['boneCount'],len(meta['nodes']))
        names=[bp.model['nodes'][i]['name'] for i in meta['nodes']]
        hips=meta['nodes'][names.index('Hips')]; spine=meta['nodes'][names.index('Spine')]; forearm=meta['nodes'][names.index('LeftForeArm')]
        # Spine Front-Back = +1 rotates Spine relative to Hips by 40 deg.
        rel0=mul(inv(q[0,hips]),q[0,spine]); rel1=mul(inv(q[1,hips]),q[1,spine])
        ang=2*np.degrees(np.arccos(np.clip(abs(float(np.dot(rel0,rel1))),0.,1.)))
        self.assertAlmostEqual(ang,40.0,places=2)
        # RootT.x += 1 translates body by humanScale while preserving relative forearm pose.
        rel_fa0=mul(inv(q[0,meta['nodes'][names.index('LeftArm')]]),q[0,forearm])
        rel_fa2=mul(inv(q[2,meta['nodes'][names.index('LeftArm')]]),q[2,forearm])
        self.assertAlmostEqual(float(np.dot(rel_fa0,rel_fa2)**2),1.0,places=5)
        self.assertTrue(np.isfinite(p).all() and np.isfinite(q).all())

    def test_motion_pipeline_defaults_to_builtin_backend_when_path_omitted(self):
        import motion_pipeline
        from scsp_motion.backend import BuiltinPoseBackend
        with tempfile.TemporaryDirectory() as t:
            td=Path(t)
            fake_info=dict(task_dir=str(td),song='mock',slots=[],mouths=[],camera=None,warnings=[],sequence_cuts=[])
            with patch('motion_pipeline.inspect_task',return_value=fake_info), \
                 patch('motion_pipeline.BuiltinPoseBackend',wraps=BuiltinPoseBackend) as spy:
                with self.assertRaises(ValueError):
                    # Timeline fails cleanly on 0 slots AFTER instantiating BuiltinPoseBackend.
                    motion_pipeline.build_task(td,td/'out',backend_path='')
                spy.assert_called_once()

    def test_hardcoded_author_credit_present_in_pmx_and_motion_exporters(self):
        root=Path(__file__).resolve().parents[1]
        pmx_src=(root/'Tools/studio_export.py').read_text(encoding='utf-8')
        mot_src=(root/'motion_pipeline.py').read_text(encoding='utf-8')
        self.assertIn('exported by scsp_mmd_tools\\nauthor:魔法少女labubu/patchouliwxy',pmx_src)
        self.assertIn('author:魔法少女labubu/patchouliwxy',mot_src)

    def test_bake_gaze_odd_60fps_cutin_avoids_rounding_oscillation(self):
        from scsp_motion.math3d import rotate
        rules=load_rules();rules['gaze_mode']='approximate'
        model=pmx_io.Model()
        for name,parent,pos in [('頭',-1,[0,10,0]),('左目',0,[0.3,10.5,-0.2]),('右目',0,[-0.3,10.5,-0.2])]:
            b=pmx_io.Bone();b.name=name;b.parent=parent;b.location=pos;model.bones.append(b)
        world={0:np.tile([0,10.,0],(8,1)),1:np.tile([0.3,10.5,-0.2],(8,1)),2:np.tile([-0.3,10.5,-0.2],(8,1))}
        # Start moving left over 8 frames (60fps), then cut in at odd 60fps frame 3 (30fps=1.5) to same side Up
        # With round(1.5)=2, old code jumped forward to frame 2's angle before frame 2 had even happened!
        events=[
            dict(eventName='ChangeTarget',startFrame=0,arg1='LeftMiddle',arg2='8'),
            dict(eventName='ChangeTarget',startFrame=3,arg1='LeftUp',arg2='6'),
        ]
        tracks,_=bake_gaze(events,model,world,{},{},None,rules)
        def eye_dir(f):
            fs,vs=tracks['左目']
            idx=int(np.searchsorted(fs,f,side='right')-1)
            return rotate(vs[max(0,idx),3:],[0.,0.,-1.])
        # At frame 2 (times=4.0), yaw must be between frame 1 (times=2.0) and frame 3 (times=6.0) monotonically,
        # and strictly less than LeftMiddle reached at frame 2 without cutin (since prev_y came from midpoint t=3.0).
        xs=[eye_dir(f)[0] for f in range(8)]
        for a,b in zip(xs,xs[1:]):
            self.assertGreaterEqual(b,a-1e-9)
        self.assertGreater(xs[2],xs[1])

    def test_partial_slot_export_solves_all_slots_for_cross_character_gaze(self):
        import motion_pipeline
        from scsp_motion import vmd
        from scsp_motion.backend import BuiltinPoseBackend
        from scsp_motion.math3d import rotate
        with tempfile.TemporaryDirectory() as t:
            root_tmp=Path(t)
            td=root_tmp/'task'
            for sub in ('animations','data','mouth','root-data'):(td/sub).mkdir(parents=True,exist_ok=True)
            def const_curve(v):return dict(m_Curve=[dict(time=0.0,value=v,inSlope=0.0,outSlope=0.0),dict(time=1.0,value=v,inSlope=0.0,outSlope=0.0)])
            for slot_idx,cid,x,z in ((1,1,0.0,0.0),(2,2,1.0,2.0)):
                (td/f'animations/test_{slot_idx:02d}.anim').write_text(f'anim{slot_idx}')
                (td/f'root-data/test_{slot_idx:02d}_root.json').write_text(json.dumps(dict(pos_x=const_curve(x),pos_z=const_curve(z))))
                (td/f'mouth/chara_{cid}_mouth.json').write_text(json.dumps(dict(frameRate=30,datas=[dict(a=0,i=0,u=0,e=0,o=0)]*31)))
            (td/'data/test_01_exp.json').write_text(json.dumps(dict(events=[dict(eventName='ChangeTarget',startFrame=0,arg1='Character02',arg2='0')])))
            (td/'data/test_02_exp.json').write_text(json.dumps(dict(events=[])))
            fake_info=dict(
                schema=1,task_dir=str(td),song='song_partial',
                slots=[
                    dict(slot=1,character_id=1,body=str(td/'animations/test_01.anim'),root=str(td/'root-data/test_01_root.json'),
                         expression=str(td/'data/test_01_exp.json'),duration=1.0),
                    dict(slot=2,character_id=2,body=str(td/'animations/test_02.anim'),root=str(td/'root-data/test_02_root.json'),
                         expression=str(td/'data/test_02_exp.json'),duration=1.0),
                ],
                mouths=[str(td/'mouth/chara_1_mouth.json'),str(td/'mouth/chara_2_mouth.json')],
                camera=None,warnings=[],sequence_cuts=[],
            )
            bp=BuiltinPoseBackend()
            cal=bp.calibrate(root_tmp/'cal_tmp')
            zero_rot=np.tile(cal['neutral_rotations'],(31,1,1))
            zero_pos=np.tile(cal['neutral_positions'],(31,1,1))
            def fake_humanoid(path,model):
                return dict(duration=1.0,frameCount=31,channels={},omittedChannels=[])
            class FakeBackend:
                model=bp.model;identity=bp.identity
                def calibrate(self,w):return cal
                def solve(self,req,w):return zero_pos.copy(),zero_rot.copy(),{}
            rules_path=root_tmp/'rules.json'
            rules_path.write_text(json.dumps({'character_aliases':{'1':'chara_1','2':'chara_2'}}))
            captured_gaze=[]
            orig_bake=bake_gaze
            def spy_bake(*args,**kwargs):
                out_gaze=orig_bake(*args,**kwargs)
                captured_gaze.append(out_gaze[0])
                return out_gaze
            with patch('motion_pipeline.inspect_task',return_value=fake_info), \
                 patch('motion_pipeline.root_curves',side_effect=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))), \
                 patch('motion_pipeline.humanoid_input',side_effect=fake_humanoid), \
                 patch('motion_pipeline.bake_gaze',side_effect=spy_bake):
                res=motion_pipeline.build_task(td,root_tmp/'out',rules_path=rules_path,slots=[1],gaze_mode='approximate',backend=FakeBackend())
            self.assertEqual(len(res['characters']),1)
            self.assertEqual(res['characters'][0]['slot'],1)
            self.assertEqual(res['characters'][0]['gaze']['unknown_targets'],{})
            self.assertTrue(Path(res['characters'][0]['file']).is_file())
            # Slot 2 is offset in front-side, so Character02 aiming steers eye horizontally in Slot 2's direction!
            basis=np.asarray(res['formation']['root_xz_to_mmd'])
            delta_mmd=np.array([1.0,2.0])@basis
            aim_vec=rotate(captured_gaze[0]['左目'][1][0,3:],[0.,0.,-1.])
            self.assertGreater(abs(aim_vec[0]),0.05)
            self.assertGreater(aim_vec[0]*delta_mmd[0],0.0)


if __name__=='__main__':unittest.main()
