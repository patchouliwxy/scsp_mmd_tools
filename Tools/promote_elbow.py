"""Promote the support Elbow to the forearm joint in an isolated working rig."""
import json
import bpy
from mathutils import Matrix


def _rows(matrix):return [list(row) for row in matrix]


def ensure_elbow_bones(rig):
    """Create missing LeftElbow/RightElbow support bones from ForeArm so downstream stages run uniformly."""
    missing = [side for side in ('Left', 'Right') if (side + 'Elbow') not in rig.data.bones]
    if not missing:
        return []
    created = []
    selected_before = tuple(bpy.context.selected_objects)
    active_before = bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for side in missing:
            forearm_name = side + 'ForeArm'
            arm_name = side + 'Arm'
            elbow_name = side + 'Elbow'
            f = rig.data.edit_bones.get(forearm_name)
            if f is None or f.parent is None or f.parent.name != arm_name:
                raise ValueError('Cannot synthesize ' + elbow_name + ' without ' + forearm_name + ' under ' + arm_name)
            e = rig.data.edit_bones.new(elbow_name)
            e.use_connect = False
            e.parent = f.parent
            e.length = max(f.length, 1e-5)
            e.matrix = f.matrix.copy()
            e.inherit_scale = f.inherit_scale
            e.use_inherit_rotation = f.use_inherit_rotation
            e.use_local_location = f.use_local_location
            e.use_deform = False
            created.append(elbow_name)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected_before:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if active_before and active_before.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = active_before
        bpy.context.view_layer.update()
    for elbow_name in created:
        rig.data.bones[elbow_name]['scsp_synthesized_from'] = elbow_name.replace('Elbow', 'ForeArm')
    return created


def promote(rig):
    if rig.get('scsp_elbow_primary'):
        raise ValueError('Elbow primary mapping already installed')
    if rig.animation_data and (rig.animation_data.action or rig.animation_data.drivers):
        raise ValueError('Animated rig needs explicit pose remapping')
    records=[]
    before={pb.name:pb.matrix_basis.copy() for pb in rig.pose.bones}
    for side in ('Left','Right'):
        f=rig.data.bones[side+'ForeArm'];e=rig.data.bones[side+'Elbow']
        if e==f or f.parent is None or e.parent!=f.parent:
            raise ValueError('Expected independent Elbow support and ForeArm under the same Arm')
        for name in (f.name,e.name):
            if rig.pose.bones[name].constraints or max(abs(before[name][i][j]-(1 if i==j else 0)) for i in range(4) for j in range(4))>1e-5:
                raise ValueError('Promotion needs neutral unconstrained lower-arm bones: '+name)
        records.append({'side':side,'source_control':f.name,'new_control':e.name,
            'old_elbow_rest':_rows(e.matrix_local),'old_forearm_rest':_rows(f.matrix_local),
            'forearm_parent':f.parent.name,'old_elbow_parent':e.parent.name,
            'reparented_children':[c.name for c in f.children]})
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for record in records:
            f=rig.data.edit_bones[record['source_control']];e=rig.data.edit_bones[record['new_control']]
            frame=f.matrix.copy();length=f.length;parent=f.parent
            children=list(f.children)
            e.use_connect=False;e.parent=parent;e.length=length;e.matrix=frame
            e.inherit_scale=f.inherit_scale;e.use_inherit_rotation=f.use_inherit_rotation;e.use_local_location=f.use_local_location
            for child in children:
                old=child.matrix.copy();child.use_connect=False;child.parent=e;child.matrix=old
            f.use_connect=False;f.parent=e;f.length=length;f.matrix=frame
    finally:bpy.ops.object.mode_set(mode='OBJECT')
    for pb in rig.pose.bones:pb.matrix_basis=before[pb.name]
    for record in records:
        f=rig.data.bones[record['source_control']];e=rig.data.bones[record['new_control']]
        e.use_deform=True;e.hide=False;e['scsp_working_role']='forearm_primary'
        e['scsp_promoted_from']=f.name;f['scsp_working_role']='retained_forearm_reference';f.hide=True
        record['new_elbow_rest']=_rows(e.matrix_local)
        record['new_elbow_parent']=e.parent.name
        record['new_forearm_parent']=f.parent.name
        record['new_children']=[c.name for c in e.children]
    aliases={r['source_control']:r['new_control'] for r in records}
    rig['scsp_elbow_primary']=json.dumps({'control_aliases':aliases,'records':records},ensure_ascii=False)
    bpy.context.view_layer.update()
    return {'control_aliases':aliases,'records':records,'bones_deleted':False,
        'policy':'Elbow adopts ForeArm rest frame/parent/children; former ForeArm is a retained reference.'}
