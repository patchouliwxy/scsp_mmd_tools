"""Modular preprocessing through skeleton cleanup and cylinder face normals."""
import argparse,json,sys
from pathlib import Path
import bpy
sys.path.insert(0,str(Path(__file__).parent))
from blender_preprocess import sha,evaluated_vertices,distance
from import_processing import prepare_materials
from eye_processing import prepare as prepare_eyes,finalize_rig as eye_pivots
from skeleton_processing import prepare as prepare_skeleton,finalize as finalize_skeleton
from body_proportions import initialize,bake_bust
from finish_preprocess import setup_preview


def run():
    p=argparse.ArgumentParser();p.add_argument('--input-blend',required=True);p.add_argument('--import-report',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.output).resolve();out.mkdir(exist_ok=True,parents=True)
    if (out/'preprocessed.blend').exists():raise FileExistsError('Use a fresh output directory')
    config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));imported=json.loads(Path(a.import_report).read_text(encoding='utf-8-sig'))
    for s in imported['sources']:
        if sha(s['path'])!=s['sha256']:raise ValueError('Source FBX changed')
    if [str(Path(x['path']).resolve()) for x in config['parts']]!=[str(Path(s['path']).resolve()) for s in imported['sources']]:raise ValueError('Source identities disagree')
    if sha(config['master_snapshot'])!=config['master_snapshot_sha256']:raise ValueError('Character master snapshot changed')
    snapshot=json.loads(Path(config['master_snapshot']).read_text(encoding='utf-8-sig'))
    if next(r for r in snapshot['records'] if r['id']==config['record']['id'])!=config['record']:raise ValueError('Character recipe changed')
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.input_blend).resolve()),load_ui=False,use_scripts=False)
    parts=[]
    for i,s in enumerate(imported['sources']):
        parts.append({'kind':s['kind'],'sourcepath':s['path'],'arm':bpy.data.objects.get(s['armature']) if s['armature'] else None,'meshes':[bpy.data.objects[m['name']] for m in s['meshes']],'objects':[o for o in bpy.data.objects if o.get('scsp_source_fbx')==s['path']],'attach_to':config['parts'][i].get('attach_to','Head')})
    report={'schema':2,'blender':bpy.app.version_string,'scope':'Modular preprocessing through part merging and MMD shape names; missing control bones deferred to PMXEditor','config_sha256':sha(a.config),'modules':{},'deferred':['eye_sheet_and_bone_review'],'shape_intermediate_restore':{'executed':False,'reason':'Not required for geometry/rig preprocessing; preserve imported source endpoint keys, defer shape-channel processing until after mesh merging'},'pipeline_steps':[]}
    report['modules']['import_fbx']=imported.get('parts_materials') if imported.get('materials_prepared') else prepare_materials(parts)
    report['pipeline_steps'].append('import_fbx_and_materials')
    if config.get('eye_texture_output_dir'):
        from eye_texture_processing import prepare as prepare_eye_textures
        report['modules']['eye_texture']=prepare_eye_textures(parts,config)
        report['pipeline_steps'].append('eye_texture')
    if config.get('skin_texture_output_dir'):
        from skin_processing import prepare as prepare_skin
        report['modules']['skin_processing']=prepare_skin(parts,config)
        report['pipeline_steps'].append('skin_processing')
    report['modules']['eye_processing']=prepare_eyes(parts,config)
    meshes=[o for part in parts for o in part['meshes']]
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'01-eyes.blend'))
    root,rig,report['modules']['skeleton_processing']=prepare_skeleton(parts,meshes)
    report['modules']['eye_processing']['rotation_pivots']=eye_pivots(rig,meshes)
    report['pipeline_steps'].extend(['eye_processing','skeleton_prepare'])
    references=bpy.data.collections.new('SCSP_SourceReferences');bpy.context.scene.collection.children.link(references)
    for part in parts:
        for obj in part['objects']:
            if obj.type!='MESH':
                for collection in list(obj.users_collection):collection.objects.unlink(obj)
                references.objects.link(obj)
    references.hide_viewport=True;references.hide_render=True
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'02-skeleton.blend'))
    report['modules']['body_proportions']=initialize(root,rig,meshes,config['parameters'],config['provenance'])
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'03-proportions.blend'))
    report['pipeline_steps'].append('body_proportions')
    report['modules']['body_proportions_bust_bake']=bake_bust(root,rig,meshes)
    report['pipeline_steps'].append('body_proportions_bust_bake')
    report['modules']['skeleton_processing']['finalize']=finalize_skeleton(rig,meshes)
    report['pipeline_steps'].extend(['skeleton_bake_apose','mmd_bone_names','prune_unused_bones'])
    if config.get('face_processing',{}).get('enabled',True):
        from face_processing import process as process_face
        report['modules']['face_processing']=process_face(meshes,config.get('face_processing'))
        report['pipeline_steps'].append('face_cylinder_custom_normals')
    if config.get('mesh_processing',{}).get('enabled',True):
        from mesh_processing import merge
        meshes,report['modules']['mesh_processing']=merge(meshes,root,rig)
        report['pipeline_steps'].append('merge_source_parts')
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'04-merged.blend'))
    if config.get('shape_processing',{}).get('enabled',True):
        if not all(o.get('scsp_merged_part') for o in meshes):raise ValueError('Shape naming requires merged source parts')
        from shape_processing import process as process_shapes
        report['modules']['shape_processing']=process_shapes(meshes)
        report['pipeline_steps'].append('rename_mmd_shape_keys')
    if config.get('export_preparation',{}).get('enabled',True):
        from export_preparation import prepare as prepare_export
        report['modules']['export_preparation']=prepare_export(root,rig,meshes)
        report['pipeline_steps'].append('export_preparation')
    report['source_fbx']=[{'path':s['path'],'sha256':s['sha256']} for s in imported['sources']]
    report['source_hashes_unchanged']=all(sha(s['path'])==s['sha256'] for s in imported['sources'])
    if not report['source_hashes_unchanged']:raise AssertionError('Source changed')
    report['mesh_summary']={'meshes':len(meshes),'vertices':sum(len(o.data.vertices) for o in meshes),'shape_frames_excluding_basis':sum(len(o.data.shape_keys.key_blocks)-1 for o in meshes if o.data.shape_keys),'shape_drivers':sum(len(o.data.shape_keys.animation_data.drivers) for o in meshes if o.data.shape_keys and o.data.shape_keys.animation_data)}
    report['working_bone_count']=len(rig.data.bones)
    report['active_shape_keys']={o.name:{k.name:float(k.value) for k in o.data.shape_keys.key_blocks[1:] if abs(k.value)>1e-7} for o in meshes if o.data.shape_keys}
    if any(report['active_shape_keys'].values()):raise AssertionError('Unexpected active expression')
    report['packed_images']=[]
    for image in bpy.data.images:
        if image.source=='FILE' and image.users and image.has_data:
            if not image.packed_file:image.pack()
            report['packed_images'].append(image.name)
    report['code_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    report['preview_setup']=setup_preview(meshes)
    report['renders_executed']=False
    if config.get('render_preview',False):
        bpy.context.scene.render.filepath=str(out/'preview.png');bpy.ops.render.render(write_still=True);report['renders_executed']=True
    root['scsp_preprocessing_report']=json.dumps(report,ensure_ascii=False);root['scsp_scope']=report['scope']
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    target=out/'preprocessed.blend';bpy.ops.wm.save_as_mainfile(filepath=str(target))
    report['output_sha256']=sha(target);report['output_blend']=str(target)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'operation-log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MODULAR_FINISHED',json.dumps({'output':str(target),'bones':report['working_bone_count'],'meshes':report['mesh_summary'],'deferred':report['deferred']},ensure_ascii=False))

if __name__=='__main__':run()
