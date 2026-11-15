"""FBX import module: source parts, textures, alpha, visibility and matte material."""
from preprocess_parts import prepare_parts,_hide_objects


def normalize_roughness(meshes,value=1.0):
    materials={m for o in meshes for m in o.data.materials if m is not None}
    result=[]
    for material in sorted(materials,key=lambda m:m.name):
        old_material=float(material.roughness);material.roughness=float(value)
        record={'material':material.name,'material_roughness_before':old_material,'material_roughness_after':float(material.roughness),'nodes':[]}
        if material.node_tree:
            for node in material.node_tree.nodes:
                if node.type!='BSDF_PRINCIPLED':continue
                socket=node.inputs.get('Roughness')
                if socket is None:continue
                before=float(socket.default_value);links=[link.from_node.name for link in socket.links]
                for link in list(socket.links):material.node_tree.links.remove(link)
                socket.default_value=float(value)
                record['nodes'].append({'node':node.name,'before':before,'after':float(socket.default_value),'removed_roughness_links':links})
        result.append(record)
    return {'policy':'Uniform matte Roughness=1.0 for material and Principled inputs','materials':result}


def refresh_visibility(meshes):
    report={'hidden_objects':[]}
    _hide_objects([{'kind':'existing_scene','objects':list(meshes)}],report)
    return report


def prepare_materials(parts):
    report=prepare_parts(parts,eye_bindings=False)
    report['roughness']=normalize_roughness([o for part in parts for o in part['meshes']])
    return report
