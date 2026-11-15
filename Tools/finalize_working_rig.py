"""Execute shoulder merge, named merged-bone cleanup, A-pose and morph naming."""
import argparse,json,sys
from pathlib import Path
import bpy
sys.path.insert(0,str(Path(__file__).parent))
from blender_preprocess import sha


def cleanup(rig,meshes,names):
    removed=[]
    for name in names:
        bone=rig.data.bones.get(name)
        if bone is None:raise ValueError('Expected merged source bone missing: '+name)
        children=[c.name for c in bone.children if c.name not in names]
        if children:raise ValueError('Merged source still parents active bones: '+name+' '+str(children))
        if rig.pose.bones[name].constraints:raise ValueError('Merged source still constrained: '+name)
        for obj in bpy.data.objects:
            if obj.parent==rig and obj.parent_type=='BONE' and obj.parent_bone==name:
                raise ValueError('Merged source still parents object: '+name)
            data=obj.animation_data
            if data:
                for curve in data.drivers:
                    for variable in curve.driver.variables:
                        for target in variable.targets:
                            if target.id==rig and (target.bone_target==name or ('["'+name+'"]') in target.data_path):
                                raise ValueError('Driver still refers to merged source: '+name)
        for mesh in meshes:
            group=mesh.vertex_groups.get(name)
            if group and any(g.group==group.index and g.weight>1e-7 for v in mesh.data.vertices for g in v.groups):
                raise ValueError('Merged bone still has positive weights: '+name)
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for name in names:
            rig.data.edit_bones.remove(rig.data.edit_bones[name]);removed.append(name)
    finally:bpy.ops.object.mode_set(mode='OBJECT')
    group_count=0
    for mesh in meshes:
        for name in names:
            group=mesh.vertex_groups.get(name)
            if group:mesh.vertex_groups.remove(group);group_count+=1
    rig['scsp_deleted_merged_sources']=json.dumps(removed)
    return {'deleted_bones':removed,'empty_vertex_groups_removed':group_count,
        'policy':'Only explicitly merged sources removed; other zero-weight control/parent bones retained.'}


def run():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--skip-morph-rename',action='store_true',help='Preserve input shape names and do not generate combinations')
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(a.input).resolve(strict=True);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    target=out/'apose.blend'
    if target.exists():raise FileExistsError('Use a new output directory')
    source_hash=sha(source);bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    rig=bpy.data.objects['SCSP_Skeleton'];root=bpy.data.objects['SCSP_Character'];meshes=[o for o in bpy.data.objects if o.type=='MESH']
    from transfer_twist_weights import _transfer
    report={'input':str(source),'input_sha256':source_hash,'face_flattening':{'executed':False,'reason':'User explicitly deferred facial processing'},'effect_validation':'Manual by user; no action regressions or render loop executed'}
    report['shoulder_merge']=_transfer(rig,meshes,(("LeftShoulder_rot","LeftShoulder"),("RightShoulder_rot","RightShoulder")), 'shoulder_rot', 'Shoulder helper weights to matching Shoulder; no normalization')
    if report['shoulder_merge']['status']!='applied':raise ValueError('Shoulder transfer failed')
    names=[side+suffix for side in ('Left','Right') for suffix in ('Shoulder_rot','Arm1_rot','Arm2_rot','Arm3_rot','ForeArm_rot','ForeArm')]
    report['cleanup']=cleanup(rig,meshes,names)
    from bake_apose import bake
    report['apose']=bake(rig,meshes,degrees=45)
    if a.skip_morph_rename:
        report['morph_names']={'status':'skipped','reason':'User requested rollback of morph naming and combination generation'}
    else:
        from rename_mmd_shapes import rename
        report['morph_names']=rename(meshes)
    report['bone_count']=len(rig.data.bones)
    report['source_unchanged']=sha(source)==source_hash
    if not report['source_unchanged']:raise AssertionError('Original input changed')
    report['pipeline_steps']=['shoulder_rot_weight_transfer','delete_explicit_merged_sources','bake_arm_45deg_apose_all_shape_keys']
    if not a.skip_morph_rename:report['pipeline_steps'].append('unity_morph_rename')
    report['code_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    root['scsp_finalize_apose_report']=json.dumps(report,ensure_ascii=False)
    if 'scsp_preprocessing_report' in root:
        previous=json.loads(root['scsp_preprocessing_report']);previous.setdefault('pipeline_steps',[]).extend(report['pipeline_steps'])
        previous['working_bone_count']=report['bone_count'];previous['rest_pose']='45 degree A-pose'
        previous['mesh_summary']={'meshes':len(meshes),'vertices':sum(len(m.data.vertices) for m in meshes),'shape_frames_excluding_basis':sum(len(m.data.shape_keys.key_blocks)-1 for m in meshes if m.data.shape_keys)}
        root['scsp_preprocessing_report']=json.dumps(previous,ensure_ascii=False)
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    report['output_sha256']=sha(target)
    (out/'operation-log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINALIZED',json.dumps({'file':str(target),'bone_count':report['bone_count'],'deleted':len(report['cleanup']['deleted_bones'])},ensure_ascii=False))

if __name__=='__main__':run()
