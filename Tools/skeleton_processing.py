"""Skeleton module: assembly, orientation, consolidation, A-pose and MMD names."""
from assemble import assemble
from bone_orientation import orient
from promote_elbow import ensure_elbow_bones, promote
from transfer_twist_weights import transfer_forearm_helpers_to_elbow,_transfer
from finalize_working_rig import cleanup
from bake_apose import bake


def _reparent_helper_children(rig,helper_targets,skip_names):
    """Reparent non-helper children (e.g. bracelets/sleeves) to the merge target before helper deletion."""
    import bpy
    pending=[]
    for src,dst in helper_targets.items():
        bone=rig.data.bones.get(src)
        if bone is None:
            continue
        for child in bone.children:
            if child.name not in skip_names:
                pending.append((child.name,src,dst))
    if not pending:
        return []
    selected_before=tuple(bpy.context.selected_objects)
    active_before=bpy.context.view_layer.objects.active
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for child_name,_src,dst in pending:
            c=rig.data.edit_bones[child_name]
            frame=c.matrix.copy()
            c.use_connect=False
            c.parent=rig.data.edit_bones[dst]
            c.matrix=frame
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected_before:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if active_before and active_before.name in bpy.data.objects:
            bpy.context.view_layer.objects.active=active_before
        bpy.context.view_layer.update()
    return [{'child':c,'from':s,'to':d} for c,s,d in pending]


def prepare(parts,meshes):
    root,rig,assembly=assemble(parts)
    synthesized_elbows=ensure_elbow_bones(rig)
    report={'assembly':assembly,'synthesized_elbows':synthesized_elbows,'orientation':orient(rig),'elbow_primary':promote(rig)}
    report['arm_weights']=transfer_forearm_helpers_to_elbow(rig,meshes)
    report['shoulder_weights']=_transfer(rig,meshes,tuple((s+'Shoulder_rot',s+'Shoulder') for s in ('Left','Right')),'shoulder_rot','Shoulder rot weights to corresponding Shoulder')
    if any(report[k]['status']!='applied' for k in ('arm_weights','shoulder_weights')):raise AssertionError('Weight consolidation failed')
    names=[s+t for s in ('Left','Right') for t in ('Shoulder_rot','Arm1_rot','Arm2_rot','Arm3_rot','ForeArm_rot','ForeArm')]
    helper_targets={s+t:s+dst for s in ('Left','Right') for t,dst in (('Shoulder_rot','Shoulder'),('Arm1_rot','Arm'),('Arm2_rot','Arm'),('Arm3_rot','Arm'),('ForeArm_rot','Elbow'),('ForeArm','Elbow'))}
    report['reparented_helper_children']=_reparent_helper_children(rig,helper_targets,set(names))
    report['cleanup']=cleanup(rig,meshes,names)
    return root,rig,report


def finalize(rig,meshes):
    # Bake after the body-proportions module so proportions are frozen once.
    report={'apose':bake(rig,meshes,degrees=45)}
    from mmd_bone_names import rename
    report['mmd_names']=rename(rig,meshes)
    report['unused_bones']=prune_unused(rig,meshes)
    return report


def prune_unused(rig,meshes):
    from prune_unused_bones import prune
    return prune(rig,meshes)


def retire_hidden_static_attachments(rig,root,meshes,declared_names=None):
    """Retire generated attachment bones used only by archived helper meshes."""
    import json
    import bpy
    from prune_unused_bones import _merge_references,prune
    if declared_names is None:
        declared_names=set()
        raw=root.get('scsp_preprocessing_report')
        if raw:
            old=json.loads(raw)
            declared_names.update(old.get('modules',{}).get('skeleton_processing',{}).get('assembly',{}).get('static_attachments',{}).values())
    declared=set(declared_names)|{b.name for b in rig.data.bones if b.get('scsp_static_attachment')}
    refs=_merge_references(rig,set(rig.data.bones.keys()),list(meshes))
    detached=[];kept=[]
    for name in sorted(declared):
        bone=rig.data.bones.get(name)
        if bone is None:continue
        attachments=[o for o in bpy.data.objects if o.parent==rig and o.parent_type=='BONE' and o.parent_bone==name]
        runtime=[r for r in refs.get(name,[]) if r['kind']!='object_bone_parent']
        useful=runtime or bone.children or any(not(o.get('scsp_reference_only') and (o.hide_render or o.hide_viewport or o.hide_get())) for o in attachments)
        if useful:
            kept.append({'bone':name,'reason':'Weights, runtime references, child bones or active objects still depend on attachment'})
            continue
        for obj in attachments:
            world=obj.matrix_world.copy();obj.parent=root;obj.parent_type='OBJECT';obj.parent_bone='';obj.matrix_world=world
            detached.append({'bone':name,'object':obj.name,'new_parent':root.name,'world_matrix':[list(r) for r in world]})
    bpy.context.view_layer.update()
    return {'detached_archived_helpers':detached,'kept_static_bones':kept,'cleanup':prune(rig,meshes),'policy':'Only generated static bones are considered for hidden-reference detachment; active static accessories keep their weighted binding'}
