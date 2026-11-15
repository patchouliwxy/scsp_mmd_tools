"""Bounded checks for a rest-axis change and iris-only eye deformation."""
import math
import bpy
from mathutils import Matrix, Vector
from blender_preprocess import evaluated_vertices, distance, mesh_fingerprint


def _clear(rig):
    for pb in rig.pose.bones: pb.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()


def _rotate(rig,name,axis,degrees):
    bpy.context.view_layer.update()
    pb=rig.pose.bones[name]
    pivot=pb.matrix.translation
    delta=Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(degrees),4,axis) @ Matrix.Translation(-pivot)
    pb.matrix=delta @ pb.matrix
    bpy.context.view_layer.update()


def _action(rig,label):
    _clear(rig)
    if label in ('raise','combined'):
        for side,sign in [('Left',-1),('Right',1)]:
            _rotate(rig,side+'Arm',Vector((0,1,0)),sign*(35 if label=='raise' else 20))
    if label in ('bend-twist','combined'):
        for side,sign in [('Left',-1),('Right',1)]:
            _rotate(rig,side+'ForeArm',Vector((0,0,1)),sign*55)
            axis=(rig.pose.bones[side+'Hand'].matrix.translation-rig.pose.bones[side+'ForeArm'].matrix.translation).normalized()
            _rotate(rig,side+'Hand',axis,sign*45)
    if label=='combined':_rotate(rig,'Head',Vector((0,0,1)),15)
    if label=='fingers':
        for side,sign in [('Left',-1),('Right',1)]:
            _rotate(rig,side+'HandIndex1',Vector((0,0,1)),sign*40)


def orient_with_checks(rig,meshes,orient):
    original_rest={b.name:b.matrix_local.copy() for b in rig.data.bones}
    raw={o.name:mesh_fingerprint(o) for o in meshes}
    initial=evaluated_vertices(meshes)
    samples=[]
    for label in ('raise','bend-twist','combined','fingers'):
        _action(rig,label)
        samples.append((label,{pb.name:pb.matrix.copy() for pb in rig.pose.bones},evaluated_vertices(meshes)))
    _clear(rig)
    result=orient(rig)
    result['neutral_mesh_error']=distance(initial,evaluated_vertices(meshes))
    result['raw_mesh_weights_shapes_unchanged']=all(mesh_fingerprint(o)==raw[o.name] for o in meshes)
    if result['neutral_mesh_error']['max_m']>1e-5 or not result['raw_mesh_weights_shapes_unchanged']:
        raise AssertionError('Orientation changed source geometry or weights')
    new_rest={b.name:b.matrix_local.copy() for b in rig.data.bones}
    trials=[]
    for label,old_pose,old_vertices in samples:
        targets={n:old_pose[n] @ original_rest[n].inverted() @ new_rest[n] for n in original_rest}
        for bone in rig.data.bones:
            kwargs={}
            if bone.parent:
                kwargs={'parent_matrix':targets[bone.parent.name],'parent_matrix_local':new_rest[bone.parent.name]}
            rig.pose.bones[bone.name].matrix_basis=bone.convert_local_to_pose(
                targets[bone.name],new_rest[bone.name],invert=True,**kwargs)
        bpy.context.view_layer.update()
        error=distance(old_vertices,evaluated_vertices(meshes))
        if error['max_m']>1e-5:raise AssertionError('World-space pose changed: '+label+' '+str(error))
        trials.append({'action':label,'mesh_error':error})
    _clear(rig)
    result['same_world_motion']=trials
    result['comparison']='P_new = P_old * inverse(R_old) * R_new; compare evaluated meshes, not equal local Euler inputs.'
    return result


def check_eye_controls(rig,meshes):
    result=[]
    base={pb.name:pb.matrix_basis.copy() for pb in rig.pose.bones}
    original=evaluated_vertices(meshes)
    for side in ('L','R'):
        source='EYE_'+side
        matches=[b.name for b in rig.data.bones if b.get('scsp_source_bone')==source]
        if len(matches)!=1:raise ValueError('Ambiguous eye bone: '+source)
        bone=matches[0]
        allowed=[o.name for o in meshes if o.get('scsp_source_object')=='eyesclera_'+side]
        if len(allowed)!=1:raise ValueError('Ambiguous iris mesh: '+side)
        try:
            rig.pose.bones[bone].matrix_basis=base[bone] @ Matrix.Rotation(math.radians(12),4,'X')
            posed=evaluated_vertices(meshes)
            movements={name:max(((a-b).length for a,b in zip(points,posed[name])),default=0) for name,points in original.items()}
            iris=max(movements[name] for name in allowed)
            other=max((v for name,v in movements.items() if name not in allowed),default=0)
            if iris<1e-5 or other>1e-5:raise AssertionError('Eye control affects wrong mesh: '+side+' '+str(movements))
            result.append({'bone':bone,'iris_mesh':allowed[0],'iris_movement_m':iris,'other_mesh_max_movement_m':other})
        finally:
            for pb in rig.pose.bones:pb.matrix_basis=base[pb.name]
            bpy.context.view_layer.update()
    return result
