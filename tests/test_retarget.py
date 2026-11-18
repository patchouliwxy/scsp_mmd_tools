"""Geometric invariants rather than asserting the converter's own formulas."""
import unittest
import numpy as np
from scsp_motion import pmx_io
from scsp_motion.config import load_rules
from scsp_motion.retarget import convert_pose,load_generic_target
from scsp_motion.math3d import rotate,world_pose


class RetargetTests(unittest.TestCase):
    def make_scene(self):
        # PMX target has an A-pose arm, a zero-offset center, and an inserted
        # identity control, while the source arm rests horizontally.
        model=pmx_io.Model()
        rows=[('全ての親',-1,[0,0,0]),('センター',0,[0,10,0]),('下半身',1,[0,10,0]),
              ('上半身',1,[0,11,0]),('左腕',3,[1,12,0]),('helper',4,[1.5,11.5,0]),
              ('左ひじ',5,[2,11,0]),('左手首',6,[3,10,0])]
        for name,parent,pos in rows:
            b=pmx_io.Bone();b.name=name;b.parent=parent;b.location=pos;model.bones.append(b)
        nodes=[dict(name='root',parent=-1,position=[0,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='Hips',parent=0,position=[0,1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='Spine',parent=1,position=[0,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftArm',parent=2,position=[.1,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftForeArm',parent=3,position=[.1,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftHand',parent=4,position=[.1,0,0],rotation=[0,0,0,1],scale=[1,1,1])]
        p,q=world_pose(nodes)
        mapping=dict(mappings=[dict(japanese=n,source_names=[s]) for n,s in [('下半身','Hips'),('上半身','Spine'),('左腕','LeftArm'),('左ひじ','LeftElbow'),('左手首','LeftHand')]])
        rules=load_rules();rules.update(source_axes=[1,1,1],units_per_meter=10)
        cal=dict(neutral_positions=p[0].tolist(),neutral_rotations=q[0].tolist())
        return model,nodes,p,q,mapping,rules,cal

    def test_bind_arm_raises_target_apose_to_source_tpose(self):
        model,nodes,p,q,mapping,rules,cal=self.make_scene()
        tracks,ik,report,world=convert_pose(model,dict(nodes=nodes),cal,p,q,np.zeros((1,3)),rules,mapping)
        arm=world[4][0];elbow=world[6][0];wrist=world[7][0]
        np.testing.assert_allclose((elbow-arm)/np.linalg.norm(elbow-arm),[1,0,0],atol=1e-9)
        np.testing.assert_allclose((wrist-elbow)/np.linalg.norm(wrist-elbow),[1,0,0],atol=1e-9)
        self.assertAlmostEqual(np.linalg.norm(elbow-arm),np.sqrt(2))

    def test_parent_rotation_preserves_lengths_and_turns_limb(self):
        model,nodes,p,q,mapping,rules,cal=self.make_scene()
        # All world bone rotations after Spine inherit a 90 degree world turn.
        turn=np.array([0,0,np.sqrt(.5),np.sqrt(.5)])
        q[:,2:]=turn
        tracks,ik,report,world=convert_pose(model,dict(nodes=nodes),cal,p,q,np.array([[3.,0,5]]),rules,mapping)
        vec=world[6][0]-world[4][0]
        np.testing.assert_allclose(vec,[0,np.sqrt(2),0],atol=1e-9)
        np.testing.assert_allclose(world[0][0],[3,0,5])

    def test_custom_mapping_replaces_default_source_assignment(self):
        model,nodes,p,q,mapping,rules,cal=self.make_scene()
        rules['bone_map']={'LeftForeArm':'helper'}
        tracks,ik,report,world=convert_pose(model,dict(nodes=nodes),cal,p,q,np.zeros((1,3)),rules,mapping)
        self.assertIn('helper',report['mapped_bones'])
        self.assertNotIn('左ひじ',report['mapped_bones'])

    def test_head_ignores_shifted_eye_pivots_and_writes_dense_neutral_eyes(self):
        model=pmx_io.Model()
        rows=[('全ての親',-1,[0,0,0]),('センター',0,[0,10,0]),('下半身',1,[0,10,0]),
              ('上半身',1,[0,11,0]),('首',3,[0,12,0]),('頭',4,[0,13,0]),
              ('左目',5,[0.3,13.9,-0.3]),('右目',5,[-0.3,13.9,-0.3]),('両目',5,[0,13.9,-1.2]),
              ('左腕',3,[1,12,0]),('左ひじ',9,[2,11,0]),('右腕',3,[-1,12,0]),('右ひじ',11,[-2,11,0]),
              ('左足',2,[0.5,10,0]),('左ひざ',13,[0.5,5,0]),('左足首',14,[0.5,1,0]),
              ('右足',2,[-0.5,10,0]),('右ひざ',16,[-0.5,5,0]),('右足首',17,[-0.5,1,0])]
        for name,parent,pos in rows:
            b=pmx_io.Bone();b.name=name;b.parent=parent;b.location=pos;model.bones.append(b)
        # Source Head -> EYE_L points up+forward (+Z), while PMX 頭 -> 左目 points up-back (-Z),
        # which previously caused a ~50 degree constant pitch bias on 頭.
        nodes=[dict(name='root',parent=-1,position=[0,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='Hips',parent=0,position=[0,1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='Spine',parent=1,position=[0,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='Neck',parent=2,position=[0,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='Head',parent=3,position=[0,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='EYE_L',parent=4,position=[.03,.09,.06],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='EYE_R',parent=4,position=[-.03,.09,.06],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftArm',parent=2,position=[.1,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftForeArm',parent=7,position=[.1,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='RightArm',parent=2,position=[-.1,.1,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='RightForeArm',parent=9,position=[-.1,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftUpLeg',parent=1,position=[.05,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftLeg',parent=11,position=[0,-.5,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='LeftFoot',parent=12,position=[0,-.4,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='RightUpLeg',parent=1,position=[-.05,0,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='RightLeg',parent=14,position=[0,-.5,0],rotation=[0,0,0,1],scale=[1,1,1]),
               dict(name='RightFoot',parent=15,position=[0,-.4,0],rotation=[0,0,0,1],scale=[1,1,1])]
        p0,q0=world_pose(nodes)
        p=np.repeat(p0,3,axis=0);q=np.repeat(q0,3,axis=0)
        # Apply a small eye rotation on frame 1 only; frame 0 and 2 remain neutral.
        q[1,5]=[0,np.sin(.05),0,np.cos(.05)]
        mapping=dict(mappings=[dict(japanese=n,source_names=[s]) for n,s in [
            ('下半身','Hips'),('上半身','Spine'),('首','Neck'),('頭','Head'),
            ('左目','EYE_L'),('右目','EYE_R'),
            ('左腕','LeftArm'),('左ひじ','LeftElbow'),('右腕','RightArm'),('右ひじ','RightElbow'),
            ('左足','LeftUpLeg'),('左ひざ','LeftLeg'),('左足首','LeftFoot'),
            ('右足','RightUpLeg'),('右ひざ','RightLeg'),('右足首','RightFoot')]])
        rules=load_rules();rules.update(source_axes=[1,1,1],units_per_meter=10)
        cal=dict(neutral_positions=p0[0].tolist(),neutral_rotations=q0[0].tolist())
        tracks,ik,report,world=convert_pose(model,dict(nodes=nodes),cal,p,q,np.zeros((3,3)),rules,mapping)
        self.assertEqual(report['rest_correction_references']['頭']['method'],'inherited_rest_frame')
        np.testing.assert_allclose(report['rest_corrections']['頭'],[0,0,0,1],atol=1e-9)
        np.testing.assert_allclose(world[6][0],[0.3,13.9,-0.3],atol=1e-9)
        self.assertEqual(len(tracks['左目'][0]),3)
        self.assertEqual(len(tracks['右目'][0]),3)
        self.assertEqual(len(tracks['両目'][0]),3)
        np.testing.assert_allclose(tracks['左目'][1][0,3:],[0,0,0,1],atol=1e-9)
        np.testing.assert_allclose(tracks['左目'][1][1,3:],q[1,5],atol=1e-9)
        np.testing.assert_allclose(tracks['両目'][1][:,3:],np.tile([0,0,0,1.],(3,1)),atol=1e-9)

    def test_generic_mmd_profile_loads_standard_apose_and_morphs(self):
        model=load_generic_target()
        self.assertEqual(model.name,'SCSP Generic MMD')
        bones={b.name for b in model.bones}
        morphs={m.name for m in model.morphs}
        for b in ('全ての親','センター','下半身','上半身','上半身3','頭','左目','右目','両目','左腕','右腕','左足ＩＫ','右足ＩＫ'):
            self.assertIn(b,bones)
        for m in ('あ','い','う','え','お','ウィンク２','ｳｨﾝｸ２右','左眉個性'):
            self.assertIn(m,morphs)


if __name__=='__main__':unittest.main()
