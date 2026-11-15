"""Finish source FBX preprocessing in Blender 5.3; no MMD or Unity runtime."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).parent))
from blender_preprocess import evaluated_vertices, distance, sha
from assemble import assemble


def _bounds(meshes):
    vertices = [v for points in evaluated_vertices(meshes).values() for v in points]
    if not vertices:
        raise ValueError('No preview geometry')
    low = Vector(tuple(min(v[i] for v in vertices) for i in range(3)))
    high = Vector(tuple(max(v[i] for v in vertices) for i in range(3)))
    return low, high


def setup_preview(meshes):
    visible = [o for o in meshes if not o.hide_render]
    low, high = _bounds(visible)
    center = (low+high)/2
    height = max(high.z-low.z, .5)
    cam_data = bpy.data.cameras.new('SCSP_Preview')
    camera = bpy.data.objects.new('SCSP_Preview', cam_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + Vector((height*.16, -height*2.6, height*.12))
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z','Y').to_euler()
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max(height*1.18, (high.x-low.x)*1.18)
    bpy.context.scene.camera = camera
    for name,offset,power,size in [('Key',(2,-3,4),500,4),('Fill',(-2,-1,2),250,3)]:
        data = bpy.data.lights.new('SCSP_'+name,'AREA')
        obj = bpy.data.objects.new('SCSP_'+name,data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = center+Vector(offset)
        obj.rotation_euler = (center-obj.location).to_track_quat('-Z','Y').to_euler()
        data.energy=power;data.shape='DISK';data.size=size
    scene=bpy.context.scene
    for engine_name in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT', 'BLENDER_WORKBENCH'):
        try:
            scene.render.engine=engine_name
            break
        except TypeError:
            continue
    scene.render.resolution_x=900;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.film_transparent=False
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.16,.18,.22,1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.6
    scene.view_settings.view_transform='Standard'
    return {'bounds_min_m':list(low),'bounds_max_m':list(high),'height_m':height}


def action_smoke(rig,meshes):
    baseline = {pb.name:pb.matrix_basis.copy() for pb in rig.pose.bones}
    modes = {pb.name:pb.rotation_mode for pb in rig.pose.bones}
    initial=evaluated_vertices(meshes)
    result=[]
    for label, actions in [('head-turn', [('Head',2,20)]),
          ('arms-bend', [('LeftArm',2,30),('RightArm',2,-30),('LeftForeArm',0,45),('RightForeArm',0,-45)])]:
        try:
            for name,axis,angle in actions:
                pb=rig.pose.bones.get(name)
                if pb is None:raise ValueError('Missing source pose bone: '+name)
                pb.rotation_mode='XYZ'
                pb.rotation_euler[axis] += math.radians(angle)
            posed=evaluated_vertices(meshes)
            finite=all(math.isfinite(c) for points in posed.values() for v in points for c in v)
            delta=distance(initial,posed)
            if not finite or delta['max_m']<1e-5:raise AssertionError('Action did not produce finite deformation: '+label)
            result.append({'action':label,'finite':finite,'max_vertex_movement_m':delta['max_m']})
        finally:
            for pb in rig.pose.bones:
                pb.rotation_mode=modes[pb.name]
                pb.matrix_basis=baseline[pb.name]
            bpy.context.view_layer.update()
    restored=distance(initial,evaluated_vertices(meshes))
    if restored['max_m']>1e-5:raise AssertionError('Smoke action restore failed')
    return {'actions':result,'restored_error':restored,
            'scope':'Responsiveness and finite geometry only; not an elbow-correction/game-runtime equivalence test.'}


def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-blend',required=True);ap.add_argument('--import-report',required=True)
    ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('preprocessed.blend','02-skeleton.blend','01-prepared.blend')):
        raise FileExistsError('Use a new output folder')
    config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    imported=json.loads(Path(a.import_report).read_text(encoding='utf-8-sig'))
    for source in imported['sources']:
        if sha(source['path'])!=source['sha256']:raise ValueError('Source FBX changed: '+source['path'])
    if [str(Path(p['path']).resolve()) for p in config['parts']] != [str(Path(s['path']).resolve()) for s in imported['sources']]:
        raise ValueError('Config and imported source identities disagree')
    snapshot=Path(config['master_snapshot'])
    if sha(snapshot)!=config['master_snapshot_sha256']:raise ValueError('Character snapshot hash mismatch')
    record=next(r for r in json.loads(snapshot.read_text(encoding='utf-8-sig'))['records'] if r['id']==config['record']['id'])
    if record!=config['record']:raise ValueError('Character record mismatch')
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.input_blend).resolve()),load_ui=False,use_scripts=False)
    parts=[]
    for i,source in enumerate(imported['sources']):
        parts.append({'kind':source['kind'],'arm':bpy.data.objects.get(source['armature']) if source['armature'] else None,
            'meshes':[bpy.data.objects[m['name']] for m in source['meshes']],
            'objects':[o for o in bpy.data.objects if o.get('scsp_source_fbx')==source['path']],
            'attach_to':config['parts'][i].get('attach_to','Head')})
    from fbx_shapes import restore_fbx_shapes
    from preprocess_parts import prepare_parts
    from body_proportions import initialize,apply,restore
    report={'schema':1,'scope':'Blender source preprocessing through skeleton assembly and body proportions only',
        'blender':bpy.app.version_string,'config_sha256':sha(a.config),
        'pipeline_steps':['source_identity','better_fbx_import','restore_inbetween_shapes','parts_materials_visibility','eye_weight_policy','iris_plane_projection','iris_rigid_retreat','eye_rim_extrusion_quad_fill','source_rig_assembly','bone_orientation','body_proportions','bounded_validation','pack_images_save'],
        'optional_steps':{'twist_weight_merge':False,'zero_weight_bone_cleanup':False},
        'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'source_fbx':[{'path':s['path'],'sha256':s['sha256']} for s in imported['sources']],
        'character_record':record,'parameter_provenance':config['provenance']}
    report['shapes']=restore_fbx_shapes(config,parts)
    report['parts']=prepare_parts(parts)
    from eye_sockets import enforce_eye_bindings,build_eye_sockets
    report['eye_bindings']=enforce_eye_bindings(parts)
    report['eye_sockets']=build_eye_sockets(parts,config.get('eye_sockets'))
    meshes=[o for p in parts for o in p['meshes']]
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'01-prepared.blend'))
    prepared=evaluated_vertices(meshes)
    root,rig,report['assembly']=assemble(parts)
    report['assembly']['neutral_mesh_error']=distance(prepared,evaluated_vertices(meshes))
    if report['assembly']['neutral_mesh_error']['max_m']>1e-5:raise AssertionError('Assembly changed neutral mesh')
    from bone_orientation import orient
    from orientation_validation import orient_with_checks,check_eye_controls
    report['orientation']=orient_with_checks(rig,meshes,orient)
    report['orientation']['prepared_mesh_error']=distance(prepared,evaluated_vertices(meshes))
    if report['orientation']['prepared_mesh_error']['max_m']>1e-5:raise AssertionError('Oriented geometry differs from prepared input')
    if config.get('eye_rotation_pivots',False):
        from eye_rotation_pivots import configure
        report['eye_rotation_pivots']=configure(rig,meshes)
        report['pipeline_steps'].insert(report['pipeline_steps'].index('body_proportions'),'eye_rotation_pivots')
    references=bpy.data.collections.new('SCSP_SourceReferences')
    bpy.context.scene.collection.children.link(references)
    for p in parts:
        for o in p['objects']:
            if o.type!='MESH':
                for coll in list(o.users_collection):coll.objects.unlink(o)
                references.objects.link(o)
    references.hide_viewport=True;references.hide_render=True
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'02-skeleton.blend'))
    report['before_proportions']=setup_preview(meshes)
    bpy.context.scene.render.filepath=str(output/'preview-before.png')
    bpy.ops.render.render(write_still=True)
    report['proportions']=initialize(root,rig,meshes,config['parameters'],config['provenance'])
    once=evaluated_vertices(meshes)
    apply(root,rig,meshes,config['parameters'])
    report['proportions_reapply_error']=distance(once,evaluated_vertices(meshes))
    if report['proportions_reapply_error']['max_m']>1e-6:raise AssertionError('Proportions accumulate')
    restore(root,rig,meshes)
    report['proportions_restore_error']=distance(prepared,evaluated_vertices(meshes))
    if report['proportions_restore_error']['max_m']>1e-5:raise AssertionError('Proportions restore failed')
    apply(root,rig,meshes,config['parameters'])
    report['actions']=action_smoke(rig,meshes)
    report['eye_control_tests']=check_eye_controls(rig,meshes)
    report['mesh_summary']={'meshes':len(meshes),'vertices':sum(len(o.data.vertices) for o in meshes),'shape_frames_excluding_basis':sum(len(o.data.shape_keys.key_blocks)-1 for o in meshes if o.data.shape_keys)}
    low,high=_bounds([o for o in meshes if not o.hide_render])
    report['after_proportions']={'bounds_min_m':list(low),'bounds_max_m':list(high),'height_m':high.z-low.z}
    report['source_hashes_unchanged']=all(sha(s['path'])==s['sha256'] for s in imported['sources'])
    if not report['source_hashes_unchanged']:raise AssertionError('Source FBX changed')
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True);bpy.context.view_layer.objects.active=rig
    # Pack referenced images so the final blend can be opened independently.
    packed=[]
    for image in bpy.data.images:
        if image.source=='FILE' and image.users and image.has_data:
            if not image.packed_file:image.pack()
            packed.append(image.name)
    report['packed_images']=packed
    # Human-readable provenance also travels with the deliverable.
    root['scsp_preprocessing_report']=json.dumps(report,ensure_ascii=False)
    root['scsp_scope']='Source preprocessing only. Original runtime auxiliary constraints not reconstructed.'
    bpy.context.scene.render.filepath=str(output/'preview.png')
    bpy.ops.render.render(write_still=True)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_distance=2.5
                area.spaces.active.region_3d.view_rotation=bpy.context.scene.camera.rotation_euler.to_quaternion()
                area.spaces.active.region_3d.view_perspective='ORTHO'
                area.spaces.active.region_3d.view_location=(low+high)/2
                area.spaces.active.clip_start=.01
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'preprocessed.blend'))
    (output/'operation-log.json').write_text(json.dumps({'schema':1,'steps':report['pipeline_steps'],'optional_steps':report['optional_steps'],'input_sources':report['source_fbx'],'config_sha256':report['config_sha256'],'code_sha256':report['code_sha256'],'output_blend':str(output/'preprocessed.blend'),'output_sha256':sha(output/'preprocessed.blend')},ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SCSP_FINISHED '+json.dumps({'blend':str(output/'preprocessed.blend'),'source_bones':report['assembly']['source_bones'],'output_bones':len(rig.data.bones),'neutral_max_m':report['assembly']['neutral_mesh_error']['max_m']}))


if __name__=='__main__':run()
