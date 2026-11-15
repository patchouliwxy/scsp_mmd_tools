"""Face shading module: transfer smooth cylinder custom corner normals only."""
import hashlib,json,math
import bpy
from mathutils import Vector
from blender_preprocess import mesh_fingerprint


def _signature(obj):
    mesh=obj.data
    state={'geometry_keys_weights':mesh_fingerprint(obj),'polygons':[list(p.vertices) for p in mesh.polygons],
        'uvs':[(u.name,[list(v.uv) for v in u.data]) for u in mesh.uv_layers],
        'materials':[m.name if m else None for m in mesh.materials],
        'shape_settings':[(k.name,k.value,k.relative_key.name) for k in mesh.shape_keys.key_blocks] if mesh.shape_keys else []}
    return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()


def process(meshes,options=None):
    options=options or {}
    matches=[o for o in meshes if o.get('scsp_source_object')=='face']
    if len(matches)!=1:raise ValueError('Expected exactly one source face mesh')
    obj=matches[0]
    if obj.get('scsp_cylinder_face_normals'):raise ValueError('Face normals already processed')
    before=_signature(obj);normals_before=[n.vector.copy() for n in obj.data.corner_normals]
    points=[obj.matrix_world@v.co for v in obj.data.vertices]
    low=Vector(tuple(min(v[i] for v in points) for i in range(3)));high=Vector(tuple(max(v[i] for v in points) for i in range(3)))
    center=(low+high)*0.5;radius=max((high.x-low.x)*0.5,(high.y-low.y)*0.5)*float(options.get('radius_scale',1.0));depth=(high.z-low.z)*float(options.get('height_scale',2.0))
    segments=int(options.get('segments',128))
    if not radius>0 or not depth>0 or segments<16:raise ValueError('Invalid cylinder parameters')
    cylinder=None;modifier=None;old_active=bpy.context.view_layer.objects.active
    try:
        bpy.ops.mesh.primitive_cylinder_add(vertices=segments,radius=radius,depth=depth,end_fill_type='NOTHING',location=center)
        cylinder=bpy.context.object;cylinder.name='SCSP_FaceNormalCylinder_TEMP'
        for poly in cylinder.data.polygons:poly.use_smooth=True
        cylinder.data.update()
        bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
        modifier=obj.modifiers.new('SCSP_CylinderCustomNormals','DATA_TRANSFER');modifier.object=cylinder
        modifier.use_loop_data=True;modifier.data_types_loops={'CUSTOM_NORMAL'};modifier.loop_mapping='POLYINTERP_NEAREST'
        modifier.mix_mode='REPLACE';modifier.mix_factor=1.0;modifier.use_object_transform=True
        # Bake only the transferred corner data. Keep every existing shape key.
        bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();evaluated=obj.evaluated_get(dg);result=evaluated.to_mesh()
        try:
            if len(result.loops)!=len(obj.data.loops) or any(a.vertex_index!=b.vertex_index for a,b in zip(result.loops,obj.data.loops)):raise AssertionError('Data transfer changed topology')
            transferred=[n.vector.copy() for n in result.corner_normals]
        finally:evaluated.to_mesh_clear()
        obj.modifiers.remove(modifier);modifier=None
        obj.data.normals_split_custom_set(transferred);obj.data.update()
    finally:
        if modifier is not None:obj.modifiers.remove(modifier)
        if cylinder is not None:
            mesh=cylinder.data;bpy.data.objects.remove(cylinder,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        if old_active and old_active.name in bpy.data.objects:bpy.context.view_layer.objects.active=old_active
    if before!=_signature(obj):raise AssertionError('Face geometry/shape keys/UV/weights/materials changed')
    after=[n.vector.copy() for n in obj.data.corner_normals]
    angles=[math.degrees(a.angle(b,0)) for a,b in zip(normals_before,after)]
    report={'mesh':obj.name,'source_object':'face','operation':'Cylinder -> Data Transfer -> Face Corner Data -> Custom Normals -> bake corner normals -> delete cylinder',
        'cylinder':{'segments':segments,'center_world':list(center),'radius_m':radius,'depth_m':depth,'caps':False,'smooth_sides':True},
        'modifier':{'type':'DATA_TRANSFER','face_corner_data':['CUSTOM_NORMAL'],'mapping':'POLYINTERP_NEAREST','mix':'REPLACE','factor':1.0},
        'bake_method':'Evaluated modifier corner normals copied to original custom normals; shape keys retained without modifier_apply mesh replacement',
        'custom_normals':obj.data.has_custom_normals,'loops':len(after),'changed_loops':sum(a>0.01 for a in angles),'mean_normal_change_degrees':sum(angles)/len(angles),'max_normal_change_degrees':max(angles),
        'geometry_shape_uv_weights_materials_unchanged':True,'before_signature':before,'after_signature':_signature(obj),'temporary_cylinder_deleted':True,
        'limitation':'Shading flattening only; no vertex flattening, mesh merging or shape-key redesign'}
    if not report['custom_normals'] or not report['changed_loops']:raise AssertionError('No face normal transfer result')
    obj['scsp_cylinder_face_normals']=json.dumps(report,ensure_ascii=False)
    return report
