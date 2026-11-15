"""Merge rendered source pieces by FBX part, preserving UVs, keys and normals."""
import json
from collections import OrderedDict,Counter
import bpy
from mathutils import Vector


def _key_settings(key):
    return {'value':float(key.value),'slider_min':float(key.slider_min),'slider_max':float(key.slider_max),'mute':bool(key.mute),'interpolation':key.interpolation}


def _merge_part(name,kind,objects,root,rig,collection):
    objects=sorted(objects,key=lambda o:o.name)
    vertices=[];edges=[];faces=[];corners=[];polygons=[];materials=[];parts=[];shape_settings={};uv_names=[];all_group_names=set()
    inverse=root.matrix_world.inverted()
    for index,obj in enumerate(objects):
        if any(m.type!='ARMATURE' for m in obj.modifiers):raise ValueError('Unhandled modifier on '+obj.name)
        for m in obj.modifiers:
            if m.object!=rig or m.use_deform_preserve_volume:raise ValueError('Unexpected skin binding')
        data=obj.data;keys=data.shape_keys
        if keys and (not keys.use_relative or keys.animation_data and (keys.animation_data.action or len(keys.animation_data.drivers))):raise ValueError('Animated/absolute shape keys require explicit handling')
        if obj.animation_data and (obj.animation_data.action or len(obj.animation_data.drivers)):raise ValueError('Animated piece cannot be merged without remapping')
        frame=inverse@obj.matrix_world;normal_frame=frame.to_3x3().inverted().transposed()
        start=len(vertices);loop_start=len(corners);poly_start=len(faces)
        base=keys.key_blocks[0].data if keys else data.vertices
        local_basis=[frame@p.co for p in base];vertices.extend(local_basis)
        edges.extend(tuple(start+i for i in e.vertices) for e in data.edges)
        material_map={}
        for i,m in enumerate(data.materials):
            if m not in materials:materials.append(m)
            material_map[i]=materials.index(m)
        faces.extend(tuple(start+i for i in p.vertices) for p in data.polygons)
        polygons.extend((material_map.get(p.material_index,0),p.use_smooth) for p in data.polygons)
        corners.extend((normal_frame@n.vector).normalized() for n in data.corner_normals)
        for uv in data.uv_layers:
            if uv.name not in uv_names:uv_names.append(uv.name)
        if keys:
            for key in keys.key_blocks[1:]:
                if key.relative_key!=keys.key_blocks[0] or key.vertex_group:raise ValueError('Relative/masked shape key needs explicit conversion: '+key.name)
                settings=_key_settings(key)
                if key.name in shape_settings and shape_settings[key.name]!=settings:raise ValueError('Conflicting shared shape key settings: '+key.name)
                shape_settings[key.name]=settings
        static_bone=obj.parent_bone if obj.parent==rig and obj.parent_type=='BONE' else None
        if static_bone and any(m.type=='ARMATURE' for m in obj.modifiers):raise ValueError('Double bound static part')
        if static_bone and static_bone not in rig.data.bones:raise ValueError('Missing attachment bone')
        all_group_names.update(g.name for g in obj.vertex_groups)
        if static_bone:
            all_group_names.add(static_bone)
            rig.data.bones[static_bone].use_deform=True
        parts.append({'object':obj,'frame':frame,'start':start,'count':len(local_basis),'loop_start':loop_start,'loop_count':len(data.loops),'polygon_start':poly_start,'polygon_count':len(data.polygons),'static_bone':static_bone,'index':index})
    mesh=bpy.data.meshes.new(name+'_Mesh');mesh.from_pydata(vertices,edges,faces);mesh.update()
    merged=bpy.data.objects.new(name,mesh);collection.objects.link(merged);merged.parent=root;merged.matrix_world=root.matrix_world.copy()
    for material in materials:mesh.materials.append(material)
    for poly,(slot,smooth) in zip(mesh.polygons,polygons):poly.material_index=slot;poly.use_smooth=smooth
    if len(mesh.loops)!=len(corners):raise AssertionError('Corner topology changed')
    # All current source materials use active UVs. Map every source active layer
    # into UV0, preserving named additional layers (hair UV1, EyeSocketUV).
    if 'UV0' not in uv_names:uv_names.insert(0,'UV0')
    for name_uv in uv_names:
        layer=mesh.uv_layers.new(name=name_uv)
        for part in parts:
            source=part['object'].data.uv_layers.get(name_uv)
            if name_uv=='UV0':source=part['object'].data.uv_layers.active or source
            if source:
                for i,point in enumerate(source.data):layer.data[part['loop_start']+i].uv=point.uv
    color_specs={}
    for part in parts:
        for attr in part['object'].data.color_attributes:
            spec=(attr.domain,attr.data_type)
            if attr.name in color_specs and color_specs[attr.name]!=spec:raise ValueError('Conflicting color layer '+attr.name)
            color_specs[attr.name]=spec
    for color_name,(domain,data_type) in color_specs.items():
        destination=mesh.color_attributes.new(name=color_name,type=data_type,domain=domain)
        for part in parts:
            source=part['object'].data.color_attributes.get(color_name)
            if source:
                offset=part['start'] if domain=='POINT' else part['loop_start']
                for i,datum in enumerate(source.data):destination.data[offset+i].color=datum.color
    edge_indices={tuple(sorted(e.vertices)):e.index for e in mesh.edges}
    for part in parts:
        source_mesh=part['object'].data;offset=part['start']
        for edge in source_mesh.edges:
            target_edge=mesh.edges[edge_indices[tuple(sorted(offset+i for i in edge.vertices))]]
            target_edge.use_seam=edge.use_seam
        for attr_name in ('sharp_edge','crease_edge','bevel_weight_edge'):
            source_attr=source_mesh.attributes.get(attr_name)
            if source_attr and source_attr.domain=='EDGE':
                target_attr=mesh.attributes.get(attr_name) or mesh.attributes.new(attr_name,source_attr.data_type,'EDGE')
                for edge in source_mesh.edges:
                    index=edge_indices[tuple(sorted(offset+i for i in edge.vertices))]
                    target_attr.data[index].value=source_attr.data[edge.index].value
    mesh.uv_layers.active_index=list(mesh.uv_layers.keys()).index('UV0')
    for layer in mesh.uv_layers:layer.active_render=layer.name=='UV0'
    groups={n:merged.vertex_groups.new(name=n) for n in sorted(all_group_names)}
    for part in parts:
        obj=part['object'];offset=part['start'];names={g.index:g.name for g in obj.vertex_groups}
        if part['static_bone']:groups[part['static_bone']].add(list(range(offset,offset+part['count'])),1.0,'REPLACE')
        else:
            for v in obj.data.vertices:
                for membership in v.groups:
                    if membership.weight>0:groups[names[membership.group]].add([offset+v.index],membership.weight,'REPLACE')
    if shape_settings:
        basis=merged.shape_key_add(name='Basis',from_mix=False);basis.value=0
        for name_key,settings in shape_settings.items():
            key=merged.shape_key_add(name=name_key,from_mix=False);key.value=0;key.relative_key=basis
            for part in parts:
                source_keys=part['object'].data.shape_keys
                old=source_keys.key_blocks.get(name_key) if source_keys else None
                if old:
                    for i,point in enumerate(old.data):key.data[part['start']+i].co=part['frame']@point.co
            for prop,value in settings.items():setattr(key,prop,value)
    mesh.update();mesh.normals_split_custom_set(corners);mesh.update()
    skin=merged.modifiers.new('SCSP_Skin','ARMATURE');skin.object=rig;skin.use_vertex_groups=True;skin.use_bone_envelopes=False
    attribute=mesh.attributes.new('scsp_source_piece','INT','POINT')
    sources=[]
    for part in parts:
        obj=part['object']
        for i in range(part['start'],part['start']+part['count']):attribute.data[i].value=part['index']
        source={'piece_id':part['index'],'object_name':obj.name,'source_object':obj.get('scsp_source_object') or ('EyeSocket_'+str(obj.get('scsp_eye_socket_side'))),'source_fbx':obj.get('scsp_source_fbx') or obj.get('scsp_eye_socket_source_fbx'),'vertex_start':part['start'],'vertex_count':part['count'],'loop_start':part['loop_start'],'loop_count':part['loop_count'],'polygon_start':part['polygon_start'],'polygon_count':part['polygon_count'],'static_attachment_bone':part['static_bone'],'shape_keys':list(obj.data.shape_keys.key_blocks.keys()) if obj.data.shape_keys else [],'matrix_to_merged':[list(r) for r in part['frame']]}
        if obj.get('scsp_cylinder_face_normals'):source['face_normals']=json.loads(obj['scsp_cylinder_face_normals'])
        sources.append(source)
    report={'mesh':merged.name,'kind':kind,'vertices':len(mesh.vertices),'polygons':len(mesh.polygons),'materials':len(materials),'shape_keys':list(mesh.shape_keys.key_blocks.keys()) if mesh.shape_keys else [],'sources':sources,'custom_normals_preserved':mesh.has_custom_normals,'uv_layers':list(mesh.uv_layers.keys()),'point_provenance_attribute':'scsp_source_piece'}
    merged['scsp_merged_part']=kind;merged['scsp_merge_sources']=json.dumps(sources,ensure_ascii=False);merged['merge_metadata']=json.dumps({'sources':sources},ensure_ascii=False);merged['scsp_source_fbx']=sources[0]['source_fbx']
    merged['scsp_source_object']='merged_'+kind
    # No welding: each source occupies an exact contiguous vertex/loop range.
    for part in parts:
        obj=part['object'];old_data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
        if old_data.users==0:bpy.data.meshes.remove(old_data)
    return merged,report


def merge(meshes,root,rig):
    meshes=list(meshes)
    from import_processing import refresh_visibility,normalize_roughness
    visibility=refresh_visibility(meshes)
    material_roughness=normalize_roughness(meshes)
    if any(o.get('scsp_merged_part') for o in meshes):raise ValueError('Parts already merged')
    hidden=[];groups=OrderedDict()
    for obj in sorted(meshes,key=lambda o:o.name):
        if obj.hide_render or obj.hide_viewport or obj.hide_get():hidden.append(obj);continue
        source=obj.get('scsp_source_fbx') or obj.get('scsp_eye_socket_source_fbx')
        if not source:raise ValueError('Source part missing: '+obj.name)
        kind=obj.name.split('__',1)[0].split('_',1)[-1] if '__' in obj.name else 'head' if obj.get('scsp_eye_socket_side') else None
        if kind not in ('head','dress','hair','acce'):raise ValueError('Unknown part kind '+str(kind))
        key=str(source).casefold()
        if key not in groups:groups[key]={'kind':kind,'objects':[]}
        groups[key]['objects'].append(obj)
    archive=bpy.data.collections.new('SCSP_HiddenMeshReferences');bpy.context.scene.collection.children.link(archive);archive.hide_viewport=True;archive.hide_render=True
    for obj in hidden:
        for collection in list(obj.users_collection):collection.objects.unlink(obj)
        archive.objects.link(obj);obj.hide_render=True;obj.hide_viewport=True;obj['scsp_reference_only']=True
    collection=bpy.data.collections.new('SCSP_MergedParts');bpy.context.scene.collection.children.link(collection)
    totals=Counter(g['kind'] for g in groups.values());indices=Counter();outputs=[];reports=[]
    for group in groups.values():
        kind=group['kind'];indices[kind]+=1
        name=kind if totals[kind]==1 else kind+'_'+str(indices[kind]).zfill(2)
        if bpy.data.objects.get(name):raise ValueError('Output name collision: '+name)
        merged,report=_merge_part(name,kind,group['objects'],root,rig,collection);outputs.append(merged);reports.append(report)
    from skeleton_processing import retire_hidden_static_attachments
    static_cleanup=retire_hidden_static_attachments(rig,root,outputs+hidden)
    return outputs,{'static_attachment_cleanup':static_cleanup,'visibility_policy':visibility,'material_roughness':material_roughness,'parts':reports,'visible_mesh_count':len(outputs),'archived_hidden_meshes':[o.name for o in hidden],'archive_collection':archive.name,'policy':'One main mesh per source FBX part; separate accessory sources remain separate; no vertex welding; hidden helper/reference meshes archived'}
