"""Source-preserving assembly into one deformation armature; no MMD conversion."""
import bpy
from mathutils import Matrix, Vector


def _neutral(arm):
    for pb in arm.pose.bones:
        if pb.constraints or max(abs(pb.matrix_basis[i][j] - (1.0 if i == j else 0.0))
                                 for i in range(4) for j in range(4)) > 1e-5:
            raise ValueError('Non-neutral/constrained source bone: ' + arm.name + '/' + pb.name)


def _rigid(matrix, label):
    loc, rot, scale = matrix.decompose()
    if min(scale) <= 0 or max(abs(s-1.0) for s in scale) > 5e-4:
        raise ValueError('Non-unit source rest frame needs explicit handling: ' + label)
    rebuilt = Matrix.LocRotScale(loc, rot, scale)
    if max(abs(matrix[i][j]-rebuilt[i][j]) for i in range(4) for j in range(4)) > 5e-4:
        raise ValueError('Sheared source rest frame: ' + label)
    return Matrix.LocRotScale(loc, rot, (1.0, 1.0, 1.0))


def _weights(obj):
    return [tuple((g.group, g.weight) for g in v.groups) for v in obj.data.vertices]


def assemble(parts):
    dresses = [p for p in parts if p['kind'] == 'dress' and p['arm']]
    if len(dresses) != 1:
        raise ValueError('Exactly one dress skeleton required')
    dress = dresses[0]['arm']
    for part in parts:
        if part['arm']:
            _neutral(part['arm'])
    master = dress.copy()
    master.data = dress.data.copy()
    master.name = 'SCSP_Skeleton'
    master.data.name = 'SCSP_SourceBones'
    bpy.context.scene.collection.objects.link(master)
    world = dress.matrix_world.copy()
    master.parent = None
    master.matrix_world = world
    master.animation_data_clear()
    master.show_in_front = True
    master.display_type = 'WIRE'
    master['scsp_role'] = 'assembled-source-skeleton'
    root = bpy.data.objects.new('SCSP_Character', None)
    root.empty_display_type = 'PLAIN_AXES'
    bpy.context.scene.collection.objects.link(root)
    mapping = {}
    records = []
    for part_index, part in enumerate(parts):
        arm = part['arm']
        if not arm:
            continue
        mapping[arm.name] = {}
        for bone in arm.data.bones:
            name = bone.name if arm == dress else f'p{part_index:02d}__{bone.name}'
            if len(name.encode('utf-8')) > 63:
                raise ValueError('Bone identity exceeds Blender name limit: ' + name)
            mapping[arm.name][bone.name] = name
    bpy.ops.object.select_all(action='DESELECT')
    master.select_set(True)
    bpy.context.view_layer.objects.active = master
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for part_index, part in enumerate(parts):
            arm = part['arm']
            if not arm or arm == dress:
                continue
            anchor = part.get('attach_to', 'Head' if part['kind'] in ('head','hair','acce') else None)
            if not anchor or anchor not in master.data.edit_bones:
                raise ValueError('Explicit attachment required: ' + part['kind'])
            for bone in arm.data.bones:
                target = mapping[arm.name][bone.name]
                if target in master.data.edit_bones:
                    raise ValueError('Ambiguous target bone: ' + target)
                frame = _rigid(world.inverted() @ arm.matrix_world @ bone.matrix_local, target)
                b = master.data.edit_bones.new(target)
                if b.name != target:
                    raise ValueError('Unexpected bone rename: ' + target)
                b.length = max(bone.length, 1e-5)
                b.matrix = frame
                b.use_deform = bone.use_deform
                b.use_connect = False
                b.use_inherit_rotation = bone.use_inherit_rotation
                b.inherit_scale = bone.inherit_scale
                b.use_local_location = bone.use_local_location
            for bone in arm.data.bones:
                b = master.data.edit_bones[mapping[arm.name][bone.name]]
                b.parent = master.data.edit_bones[mapping[arm.name][bone.parent.name] if bone.parent else anchor]
        # Static meshes stay static in the imported snapshot and acquire an explicit attachment here.
        static = []
        for part_index, part in enumerate(parts):
            for obj in part['meshes']:
                mods = [m for m in obj.modifiers if m.type == 'ARMATURE']
                if mods:
                    continue
                name = f'p{part_index:02d}__static_{len(static):03d}'
                b = master.data.edit_bones.new(name)
                b.head = world.inverted() @ obj.matrix_world.translation
                b.tail = b.head + Vector((0, 0.01, 0))
                b.parent = master.data.edit_bones['Head'] if part['kind'] in ('head','hair','acce') else master.data.edit_bones['root']
                b.use_deform = False
                b['scsp_static_attachment']=True
                b['scsp_static_source_object']=obj.name
                static.append((obj,name))
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    static_names = {obj.name: bone for obj,bone in static}
    for part in parts:
        arm = part['arm']
        for obj in part['meshes']:
            old_world = obj.matrix_world.copy()
            mods = [m for m in obj.modifiers if m.type == 'ARMATURE']
            if mods:
                if len(mods) != 1 or not arm or mods[0].object != arm or not mods[0].show_viewport:
                    raise ValueError('Ambiguous skin modifier: ' + obj.name)
                original_weights = _weights(obj)
                names_before = [g.name for g in obj.vertex_groups]
                positive = {g.group for v in obj.data.vertices for g in v.groups if g.weight > 1e-8}
                for group_index in positive:
                    group = obj.vertex_groups[group_index]
                    if group.name not in mapping[arm.name]:
                        raise ValueError('Unresolved positive skin group: ' + obj.name + '/' + group.name)
                    if not arm.data.bones[group.name].use_deform:
                        raise ValueError('Positive weights on disabled deform bone: ' + obj.name + '/' + group.name)
                for group in obj.vertex_groups:
                    if group.name in mapping[arm.name]:
                        group.name = mapping[arm.name][group.name]
                mods[0].object = master
                if obj.parent_type == 'BONE':
                    raise ValueError('Unexpected bone parenting on skinned mesh: ' + obj.name)
                obj.parent = root
                obj.matrix_world = old_world
                if _weights(obj) != original_weights:
                    raise AssertionError('Vertex weights changed: ' + obj.name)
                records.append({'mesh':obj.name,'source_armature':arm.name,'target_armature':master.name,
                    'group_mapping':dict(zip(names_before,[g.name for g in obj.vertex_groups])),
                    'weights_exact':True})
            else:
                obj.parent = master
                obj.parent_type = 'BONE'
                obj.parent_bone = static_names[obj.name]
                bpy.context.view_layer.update()
                obj.matrix_world = old_world
        if arm:
            arm['scsp_role'] = 'source-reference-only'
            arm.hide_set(True)
            arm.hide_render = True
    master.parent = root
    master.matrix_world = world
    # Source reference hierarchy remains available, but is outside the output root.
    bpy.context.view_layer.update()
    maximum = 0.0
    for part in parts:
        arm = part['arm']
        if not arm:
            continue
        for bone in arm.data.bones:
            target = master.data.bones[mapping[arm.name][bone.name]]
            old = arm.matrix_world @ bone.matrix_local
            new = master.matrix_world @ target.matrix_local
            maximum = max(maximum, max(abs(old[i][j]-new[i][j]) for i in range(4) for j in range(4)))
            target['scsp_source_armature'] = arm.name
            target['scsp_source_bone'] = bone.name
    if maximum > 1e-4:
        raise AssertionError('Rest frame changed: ' + str(maximum))
    master['scsp_source_mapping'] = __import__('json').dumps(mapping,ensure_ascii=False)
    return root, master, {'source_bones':sum(len(p['arm'].data.bones) for p in parts if p['arm']),
        'output_bones':len(master.data.bones),'rest_matrix_max_abs_error':maximum,
        'skin_bindings':records,'static_attachments':static_names,
        'policy':'Keep source bone identities and weights; append part-specific branches with explicit Head attachment. No same-name collapse, rest reorientation, auxiliary bone deletion, or MMD controls.',
        'limitations':['Game runtime auxiliary constraints absent from FBX are not reconstructed.']}
