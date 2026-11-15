"""Place rigid planar iris rotation centers behind the iris; no topology edits."""
import argparse,json,math,sys
from pathlib import Path
import bpy
from mathutils import Vector,Matrix
sys.path.insert(0,str(Path(__file__).parent))
from blender_preprocess import sha


def configure(rig, meshes):
    if rig.mode != 'OBJECT':
        raise ValueError('Eye pivot configuration requires Object mode')
    if rig.get('scsp_eye_rotation_pivots'):
        raise ValueError('Eye pivots already configured')
    records = []
    inverse = rig.matrix_world.inverted()
    for side in ('L', 'R'):
        bones = [b for b in rig.data.bones if b.get('scsp_source_bone') == 'EYE_' + side]
        sheets = [o for o in meshes if o.get('scsp_source_object') == 'eyesclera_' + side]
        if len(bones) != 1 or len(sheets) != 1:
            raise ValueError('Ambiguous eye metadata ' + side)
        bone = bones[0]
        sheet = sheets[0]
        if bone.children or rig.pose.bones[bone.name].constraints:
            raise ValueError('Eye bone has children/constraints')
        if max(abs(rig.pose.bones[bone.name].matrix_basis[r][c] - Matrix.Identity(4)[r][c]) for r in range(4) for c in range(4)) > 1e-6:
            raise ValueError('Eye pose is not neutral')
        points = [sheet.matrix_world @ v.co for v in sheet.data.vertices]
        center = sum(points, Vector()) / len(points)
        stored_pivot = sheet.get('scsp_eye_pivot_world')
        stored_forward = sheet.get('scsp_eye_forward_world')
        if stored_pivot is not None and stored_forward is not None:
            pivot = Vector(stored_pivot)
            eye_forward = Vector(stored_forward).normalized()
            radius = float(sheet.get('scsp_iris_rim_radius', (center - pivot).length))
            tail = pivot + eye_forward * radius
            rule = 'Spherical concave iris pivot aligned to eye-hole sphere center; world front -Y; rigid rotation +-20 deg'
        else:
            radius = (max(v.x for v in points) - min(v.x for v in points)) * 0.5
            if radius <= 0:
                raise ValueError('Invalid iris width')
            pivot = center + Vector((0, radius, 0))
            tail = center
            rule = 'Fallback behind iris by half iris mesh world-X width'
        records.append({
            'side': side,
            'bone': bone.name,
            'iris': sheet.name,
            'old_head': list(bone.head_local),
            'old_tail': list(bone.tail_local),
            'iris_center_world': list(center),
            'pivot_world': list(pivot),
            'radius_world_m': radius,
            'new_head': list(inverse @ pivot),
            'new_tail': list(inverse @ tail),
            'rule': rule,
        })
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for r in records:
            b = rig.data.edit_bones[r['bone']]
            b.head = r['new_head']
            b.tail = r['new_tail']
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    report = {
        'eyes': records,
        'geometry_changed': False,
        'shape_keys_changed': False,
        'weights_changed': False,
        'controls': 'Pose mode, local X for up/down and local Z for left/right within +-20 deg; local Y is roll.',
    }
    rig['scsp_eye_rotation_pivots'] = json.dumps(report, ensure_ascii=False)
    return report


def _positions(meshes):
    bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();out={}
    for o in meshes:
        e=o.evaluated_get(dg);m=e.to_mesh()
        out[o.name]=[e.matrix_world@v.co for v in m.vertices];e.to_mesh_clear()
    return out


def run():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(a.input).resolve(strict=True);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);target=out/'eye-rotation.blend'
    if target.exists():raise FileExistsError(target)
    source_hash=sha(source);bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    rig=bpy.data.objects['SCSP_Skeleton'];meshes=[o for o in bpy.data.objects if o.type=='MESH']
    before=_positions(meshes);report=configure(rig,meshes);after=_positions(meshes)
    report['neutral_max_error_m']=max((u-v).length for n,ps in before.items() for u,v in zip(ps,after[n]))
    if report['neutral_max_error_m']>1e-6:raise AssertionError('Neutral geometry changed')
    report['input']=str(source);report['input_sha256']=source_hash
    report['code_sha256']=sha(Path(__file__))
    # Bounded numeric control check; no render, no pose saved.
    report['rotation_checks']=[]
    for rec in report['eyes']:
        pb=rig.pose.bones[rec['bone']];basis=pb.matrix_basis.copy()
        for axis in ('X','Z'):
            pb.matrix_basis=basis@Matrix.Rotation(math.radians(15),4,axis)
            moved=_positions(meshes)
            iris=rec['iris'];other_error=max(((u-v).length for n,ps in after.items() if n!=iris for u,v in zip(ps,moved[n])),default=0)
            iris_error=max((u-v).length for u,v in zip(after[iris],moved[iris]))
            if iris_error<1e-4 or other_error>1e-6:raise AssertionError('Eye rotation isolation failed')
            report['rotation_checks'].append({'side':rec['side'],'local_axis':axis,'degrees':15,'iris_max_motion_m':iris_error,'non_iris_max_error_m':other_error})
            pb.matrix_basis=basis;bpy.context.view_layer.update()
    if sha(source)!=source_hash:raise AssertionError('Source changed')
    root=rig.parent;root['scsp_eye_rotation_report']=json.dumps(report,ensure_ascii=False)
    if 'scsp_preprocessing_report' in root:
        old=json.loads(root['scsp_preprocessing_report']);old.setdefault('pipeline_steps',[]).append('eye_rotation_pivots');old['eye_rotation_pivots']=report;root['scsp_preprocessing_report']=json.dumps(old,ensure_ascii=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(target));report['output_sha256']=sha(target)
    (out/'operation-log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EYE_PIVOTS',json.dumps({'output':str(target),'neutral_max_error_m':report['neutral_max_error_m'],'checks':report['rotation_checks']},ensure_ascii=False))

if __name__=='__main__':run()
