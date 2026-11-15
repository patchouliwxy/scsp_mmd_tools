"""Clean-room MIT ufbx-backed FBX importer for Blender 5.3.

Loads FBX files via Tools/ufbx_bridge.dll (compiled from ThirdParty/ufbx)
and constructs Blender Armature, Empty hierarchy, and Mesh objects with
exact geometry, custom loop normals, all UV sets, skin weights, shape keys,
and Principled BSDF materials.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable

import bpy
import numpy as np
from mathutils import Matrix, Vector

_DEFAULT_DLL_PATH = Path(__file__).resolve().parent / "ufbx_bridge.dll"

_c_int32 = ctypes.c_int32
_c_float = ctypes.c_float
_c_char_p = ctypes.c_char_p
_c_void_p = ctypes.c_void_p
_c_int32_p = ctypes.POINTER(_c_int32)
_c_float_p = ctypes.POINTER(_c_float)

_LIB_CACHE: dict[Path, ctypes.CDLL] = {}


def _load_bridge(dll_path: Path | None = None) -> ctypes.CDLL:
    """Load and configure ctypes signatures for ufbx_bridge.dll."""
    path = (dll_path or _DEFAULT_DLL_PATH).resolve()
    cached = _LIB_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise FileNotFoundError(f"ufbx bridge DLL not found: {path}")

    lib = ctypes.CDLL(str(path))
    lib.scsp_fbx_load.argtypes = [_c_char_p, ctypes.c_char_p, _c_int32]
    lib.scsp_fbx_load.restype = _c_void_p

    lib.scsp_fbx_free.argtypes = [_c_void_p]
    lib.scsp_fbx_free.restype = None

    lib.scsp_fbx_get_armature_name.argtypes = [_c_void_p]
    lib.scsp_fbx_get_armature_name.restype = _c_char_p

    lib.scsp_fbx_get_counts.argtypes = [_c_void_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p]
    lib.scsp_fbx_get_counts.restype = None

    lib.scsp_fbx_get_node_info.argtypes = [
        _c_void_p, _c_int32, _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_float_p
    ]
    lib.scsp_fbx_get_node_info.restype = _c_char_p

    lib.scsp_fbx_get_bone_info.argtypes = [_c_void_p, _c_int32, _c_int32_p, _c_float_p, _c_float_p]
    lib.scsp_fbx_get_bone_info.restype = _c_char_p

    lib.scsp_fbx_get_material_info.argtypes = [
        _c_void_p, _c_int32, _c_float_p, ctypes.POINTER(_c_char_p), ctypes.POINTER(_c_char_p)
    ]
    lib.scsp_fbx_get_material_info.restype = _c_char_p

    lib.scsp_fbx_get_mesh_header.argtypes = [
        _c_void_p, _c_int32,
        _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p,
        _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p,
    ]
    lib.scsp_fbx_get_mesh_header.restype = _c_char_p

    lib.scsp_fbx_fill_mesh_geometry.argtypes = [
        _c_void_p, _c_int32,
        _c_float_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_int32_p, _c_float_p, _c_int32_p,
    ]
    lib.scsp_fbx_fill_mesh_geometry.restype = None

    lib.scsp_fbx_fill_mesh_uv.argtypes = [_c_void_p, _c_int32, _c_int32, _c_float_p]
    lib.scsp_fbx_fill_mesh_uv.restype = _c_char_p

    lib.scsp_fbx_fill_mesh_skin.argtypes = [_c_void_p, _c_int32, _c_int32_p, _c_int32_p, _c_float_p]
    lib.scsp_fbx_fill_mesh_skin.restype = None

    lib.scsp_fbx_get_shape_header.argtypes = [_c_void_p, _c_int32, _c_int32, _c_int32_p]
    lib.scsp_fbx_get_shape_header.restype = _c_char_p

    lib.scsp_fbx_fill_shape_offsets.argtypes = [_c_void_p, _c_int32, _c_int32, _c_int32_p, _c_float_p]
    lib.scsp_fbx_fill_shape_offsets.restype = None

    _LIB_CACHE[path] = lib
    return lib


def _mat4_from_c(flat16: Iterable[float]) -> Matrix:
    """Convert row-major 16-float C array to Blender mathutils.Matrix."""
    vals = list(flat16)
    return Matrix((
        vals[0:4],
        vals[4:8],
        vals[8:12],
        vals[12:16],
    ))


def _resolve_texture_path(fbx_dir: Path, tex_filename: str, tex_rel: str) -> Path | None:
    """Resolve an FBX material texture path using AssetStudio layout conventions."""
    candidates: list[Path] = []
    for raw in (tex_rel, tex_filename):
        if not raw:
            continue
        norm = raw.replace("\\", "/")
        p = Path(norm)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(fbx_dir / norm)
            candidates.append(fbx_dir.parent / norm)
        base = Path(norm).name
        if base:
            candidates.extend([
                fbx_dir / base,
                fbx_dir / "Textures" / base,
                fbx_dir / "Texture2D" / base,
                fbx_dir.parent / "Textures" / base,
                fbx_dir.parent / "Texture2D" / base,
                fbx_dir.parent / base,
            ])
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _build_material(
    name: str,
    rgba: tuple[float, float, float, float],
    tex_path: Path | None,
) -> bpy.types.Material:
    """Create a Principled BSDF Blender material matching BetterFBX import settings."""
    mat = bpy.data.materials.new(name=name or "Material")
    mat.use_nodes = True
    mat.diffuse_color = rgba
    mat.use_backface_culling = False

    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return mat

    base_inp = bsdf.inputs.get("Base Color")
    if base_inp is None:
        return mat

    base_inp.default_value = rgba
    if tex_path is not None:
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.location = (-300, 300)
        tex_node.image = bpy.data.images.load(str(tex_path), check_existing=True)
        nt.links.new(tex_node.outputs["Color"], base_inp)
    return mat


def _set_object_parent(
    obj: bpy.types.Object,
    parent_obj: bpy.types.Object | None,
    world_mat: Matrix,
    parent_world_mat: Matrix | None = None,
) -> None:
    """Assign parent with exact local basis derived from explicit parent world matrix."""
    if parent_obj is not None:
        obj.parent = parent_obj
        obj.matrix_parent_inverse = Matrix.Identity(4)
        p_inv = (parent_world_mat if parent_world_mat is not None else parent_obj.matrix_world).inverted_safe()
        obj.matrix_basis = p_inv @ world_mat
    else:
        obj.matrix_world = world_mat


def _build_armature(
    lib: ctypes.CDLL,
    handle: int,
    num_bones: int,
    arm_name: str,
    collection: bpy.types.Collection,
) -> tuple[bpy.types.Object, list[str]]:
    """Create Blender Armature object (world Identity) + Z-up EditBones."""
    arm_data = bpy.data.armatures.new(name=arm_name or "Armature")
    arm_obj = bpy.data.objects.new(name=arm_name or "Armature", object_data=arm_data)
    collection.objects.link(arm_obj)
    arm_obj.matrix_world = Matrix.Identity(4)

    bone_records: list[tuple[str, int, Matrix, float]] = []
    parent_idx = _c_int32()
    tail_len = _c_float()
    mat_buf = (_c_float * 16)()
    for bi in range(num_bones):
        b_name_b = lib.scsp_fbx_get_bone_info(
            handle, bi, ctypes.byref(parent_idx), mat_buf, ctypes.byref(tail_len)
        )
        b_name = b_name_b.decode("utf-8", errors="replace") if b_name_b else f"Bone_{bi}"
        bind_world = _mat4_from_c(mat_buf)
        bone_records.append((b_name, int(parent_idx.value), bind_world, max(1e-4, float(tail_len.value))))

    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: list[bpy.types.EditBone] = []
    bone_names: list[str] = []
    try:
        for b_name, _, bind_world, t_len in bone_records:
            rot3 = bind_world.to_3x3().normalized()
            head = bind_world.translation
            y_axis = Vector(rot3.col[1]).normalized()
            z_axis = Vector(rot3.col[2]).normalized()
            eb = arm_data.edit_bones.new(b_name)
            eb.head = head
            eb.tail = head + y_axis * t_len
            eb.align_roll(z_axis)
            eb.use_connect = False
            edit_bones.append(eb)
            bone_names.append(eb.name)
        for bi, (_, p_bi, _, _) in enumerate(bone_records):
            if 0 <= p_bi < len(edit_bones):
                edit_bones[bi].parent = edit_bones[p_bi]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = prev_active

    return arm_obj, bone_names


def _build_mesh_object(
    lib: ctypes.CDLL,
    handle: int,
    mi: int,
    scene_mats: list[bpy.types.Material],
    bone_names: list[str],
    arm_obj: bpy.types.Object | None,
    node_objects: dict[int, bpy.types.Object],
    node_world_mats: dict[int, Matrix],
    nodes_meta: list[tuple[str, int, int, int, int, Matrix]],
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create a Blender Mesh object with custom normals, UVs, weights, and shape keys."""
    node_idx = _c_int32()
    num_verts = _c_int32()
    num_polys = _c_int32()
    num_loops = _c_int32()
    num_uv_layers = _c_int32()
    num_mats = _c_int32()
    num_skin_weights = _c_int32()
    num_shapes = _c_int32()
    is_skinned = _c_int32()
    has_normals = _c_int32()

    mesh_name_b = lib.scsp_fbx_get_mesh_header(
        handle, mi,
        ctypes.byref(node_idx),
        ctypes.byref(num_verts),
        ctypes.byref(num_polys),
        ctypes.byref(num_loops),
        ctypes.byref(num_uv_layers),
        ctypes.byref(num_mats),
        ctypes.byref(num_skin_weights),
        ctypes.byref(num_shapes),
        ctypes.byref(is_skinned),
        ctypes.byref(has_normals),
    )
    ni = int(node_idx.value)
    node_name, _, parent_ni, _, _, world_mat = nodes_meta[ni]
    mesh_name = mesh_name_b.decode("utf-8", errors="replace") if mesh_name_b else node_name

    nv = int(num_verts.value)
    np_ = int(num_polys.value)
    nl = int(num_loops.value)
    nm = int(num_mats.value)

    verts_xyz = np.empty(nv * 3, dtype=np.float32) if nv > 0 else np.empty(0, dtype=np.float32)
    loop_start = np.empty(np_, dtype=np.int32) if np_ > 0 else np.empty(0, dtype=np.int32)
    loop_total = np.empty(np_, dtype=np.int32) if np_ > 0 else np.empty(0, dtype=np.int32)
    poly_mat_idx = np.empty(np_, dtype=np.int32) if np_ > 0 else np.empty(0, dtype=np.int32)
    loop_verts = np.empty(nl, dtype=np.int32) if nl > 0 else np.empty(0, dtype=np.int32)
    loop_normals = np.empty(nl * 3, dtype=np.float32) if nl > 0 else np.empty(0, dtype=np.float32)
    mesh_scene_mats = np.empty(nm, dtype=np.int32) if nm > 0 else np.empty(0, dtype=np.int32)

    lib.scsp_fbx_fill_mesh_geometry(
        handle, mi,
        verts_xyz.ctypes.data_as(_c_float_p) if nv > 0 else None,
        loop_start.ctypes.data_as(_c_int32_p) if np_ > 0 else None,
        loop_total.ctypes.data_as(_c_int32_p) if np_ > 0 else None,
        poly_mat_idx.ctypes.data_as(_c_int32_p) if np_ > 0 else None,
        loop_verts.ctypes.data_as(_c_int32_p) if nl > 0 else None,
        loop_normals.ctypes.data_as(_c_float_p) if nl > 0 else None,
        mesh_scene_mats.ctypes.data_as(_c_int32_p) if nm > 0 else None,
    )

    me = bpy.data.meshes.new(name=mesh_name)
    if nv > 0:
        me.vertices.add(nv)
        me.vertices.foreach_set("co", verts_xyz)
    if nl > 0:
        me.loops.add(nl)
        me.loops.foreach_set("vertex_index", loop_verts)
    if np_ > 0:
        me.polygons.add(np_)
        me.polygons.foreach_set("loop_start", loop_start)
        me.polygons.foreach_set("loop_total", loop_total)
        me.polygons.foreach_set("use_smooth", np.ones(np_, dtype=bool))
        if nm > 0:
            me.polygons.foreach_set("material_index", np.clip(poly_mat_idx, 0, nm - 1))

    for sm_idx in mesh_scene_mats:
        if 0 <= int(sm_idx) < len(scene_mats):
            me.materials.append(scene_mats[int(sm_idx)])
        else:
            me.materials.append(None)

    me.validate(verbose=False, clean_customdata=False)
    me.update()

    if int(has_normals.value) and nl > 0 and len(me.loops) == nl:
        me.normals_split_custom_set(loop_normals.reshape((nl, 3)))

    nuv = int(num_uv_layers.value)
    if nuv > 0 and nl > 0:
        uv_buf = np.empty(nl * 2, dtype=np.float32)
        for u in range(nuv):
            uv_name_b = lib.scsp_fbx_fill_mesh_uv(
                handle, mi, u, uv_buf.ctypes.data_as(_c_float_p)
            )
            uv_name = uv_name_b.decode("utf-8", errors="replace") if uv_name_b else ""
            if not uv_name:
                uv_name = "UVMap" if u == 0 else f"UVMap.{u:03d}"
            uv_layer = me.uv_layers.new(name=uv_name)
            if len(uv_layer.data) == nl:
                uv_layer.data.foreach_set("uv", uv_buf)

    obj = bpy.data.objects.new(name=node_name, object_data=me)
    collection.objects.link(obj)
    node_objects[ni] = obj
    node_world_mats[ni] = world_mat

    if int(is_skinned.value) and arm_obj is not None:
        _set_object_parent(obj, arm_obj, world_mat, Matrix.Identity(4))
        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj
    else:
        parent_obj = node_objects.get(parent_ni)
        _set_object_parent(obj, parent_obj, world_mat, node_world_mats.get(parent_ni))

    nsw = int(num_skin_weights.value)
    if nsw > 0 and bone_names:
        b_idx = np.empty(nsw, dtype=np.int32)
        v_idx = np.empty(nsw, dtype=np.int32)
        weights = np.empty(nsw, dtype=np.float32)
        lib.scsp_fbx_fill_mesh_skin(
            handle, mi,
            b_idx.ctypes.data_as(_c_int32_p),
            v_idx.ctypes.data_as(_c_int32_p),
            weights.ctypes.data_as(_c_float_p),
        )
        vgroups: dict[int, bpy.types.VertexGroup] = {}
        for k in range(nsw):
            bi = int(b_idx[k])
            vi = int(v_idx[k])
            w = float(weights[k])
            if bi < 0 or bi >= len(bone_names) or vi < 0 or vi >= nv or w <= 0.0:
                continue
            vg = vgroups.get(bi)
            if vg is None:
                vg = obj.vertex_groups.new(name=bone_names[bi])
                vgroups[bi] = vg
            vg.add((vi,), w, "ADD")

    nsh = int(num_shapes.value)
    if nsh > 0 and nv > 0:
        base_coords = verts_xyz.reshape((nv, 3))
        obj.shape_key_add(name="Basis", from_mix=False)
        me.shape_keys.use_relative = True
        off_cnt = _c_int32()
        for si in range(nsh):
            s_name_b = lib.scsp_fbx_get_shape_header(handle, mi, si, ctypes.byref(off_cnt))
            s_name = s_name_b.decode("utf-8", errors="replace") if s_name_b else f"Shape_{si}"
            no = int(off_cnt.value)
            sk = obj.shape_key_add(name=s_name, from_mix=False)
            sk.value = 0.0
            if no > 0:
                s_vi = np.empty(no, dtype=np.int32)
                s_dl = np.empty(no * 3, dtype=np.float32)
                lib.scsp_fbx_fill_shape_offsets(
                    handle, mi, si,
                    s_vi.ctypes.data_as(_c_int32_p),
                    s_dl.ctypes.data_as(_c_float_p),
                )
                coords = base_coords.copy()
                valid = (s_vi >= 0) & (s_vi < nv)
                np.add.at(coords, s_vi[valid], s_dl.reshape((no, 3))[valid])
                sk.data.foreach_set("co", coords.ravel())

    return obj


def import_fbx_file(
    filepath: str | Path,
    collection: bpy.types.Collection | None = None,
    dll_path: Path | None = None,
) -> list[bpy.types.Object]:
    """Import an FBX file into Blender using the clean-room ufbx bridge."""
    path = Path(filepath).resolve()
    lib = _load_bridge(dll_path)

    err_buf = ctypes.create_string_buffer(512)
    handle = lib.scsp_fbx_load(str(path).encode("utf-8"), err_buf, len(err_buf))
    if not handle:
        msg = err_buf.value.decode("utf-8", errors="replace") or "unknown ufbx error"
        raise RuntimeError(f"ufbx_bridge failed to load {path}: {msg}")

    if collection is None:
        collection = bpy.context.scene.collection

    created: list[bpy.types.Object] = []
    try:
        c_nodes, c_bones, c_meshes, c_mats = _c_int32(), _c_int32(), _c_int32(), _c_int32()
        lib.scsp_fbx_get_counts(
            handle,
            ctypes.byref(c_nodes),
            ctypes.byref(c_bones),
            ctypes.byref(c_meshes),
            ctypes.byref(c_mats),
        )
        n_nodes = int(c_nodes.value)
        n_bones = int(c_bones.value)
        n_meshes = int(c_meshes.value)
        n_mats = int(c_mats.value)

        arm_name_b = lib.scsp_fbx_get_armature_name(handle)
        arm_name = arm_name_b.decode("utf-8", errors="replace") if arm_name_b else "Armature"

        # 1. Read full node table
        nodes_meta: list[tuple[str, int, int, int, int, Matrix]] = []
        arm_node_idx = -1
        c_kind, c_parent, c_midx, c_bidx = _c_int32(), _c_int32(), _c_int32(), _c_int32()
        mat_buf = (_c_float * 16)()
        for ni in range(n_nodes):
            n_name_b = lib.scsp_fbx_get_node_info(
                handle, ni,
                ctypes.byref(c_kind),
                ctypes.byref(c_parent),
                ctypes.byref(c_midx),
                ctypes.byref(c_bidx),
                mat_buf,
            )
            n_name = n_name_b.decode("utf-8", errors="replace") if n_name_b else f"Node_{ni}"
            kind = int(c_kind.value)
            if kind == 3 and arm_node_idx < 0:
                arm_node_idx = ni
            nodes_meta.append((
                n_name, kind, int(c_parent.value), int(c_midx.value), int(c_bidx.value), _mat4_from_c(mat_buf)
            ))

        # 2. Build materials
        fbx_dir = path.parent
        scene_mats: list[bpy.types.Material] = []
        rgba_buf = (_c_float * 4)()
        tex_fn = _c_char_p()
        tex_rel = _c_char_p()
        for mi in range(n_mats):
            m_name_b = lib.scsp_fbx_get_material_info(
                handle, mi, rgba_buf, ctypes.byref(tex_fn), ctypes.byref(tex_rel)
            )
            m_name = m_name_b.decode("utf-8", errors="replace") if m_name_b else f"Material_{mi}"
            rgba = (float(rgba_buf[0]), float(rgba_buf[1]), float(rgba_buf[2]), float(rgba_buf[3]))
            fn_s = tex_fn.value.decode("utf-8", errors="replace") if tex_fn.value else ""
            rel_s = tex_rel.value.decode("utf-8", errors="replace") if tex_rel.value else ""
            tex_path = _resolve_texture_path(fbx_dir, fn_s, rel_s)
            scene_mats.append(_build_material(m_name, rgba, tex_path))

        # 3. Build Armature (always world Identity so EditBones stay Z-up world coordinates)
        node_objects: dict[int, bpy.types.Object] = {}
        node_world_mats: dict[int, Matrix] = {}
        arm_obj: bpy.types.Object | None = None
        bone_names: list[str] = []
        if n_bones > 0:
            arm_obj, bone_names = _build_armature(
                lib, handle, n_bones, arm_name, collection
            )
            created.append(arm_obj)
            if arm_node_idx >= 0:
                node_objects[arm_node_idx] = arm_obj
                node_world_mats[arm_node_idx] = Matrix.Identity(4)

        # 4. Build Empty hierarchy (kind == 0) in parent-first passes
        empty_indices = [ni for ni, meta in enumerate(nodes_meta) if meta[1] == 0]
        remaining = list(empty_indices)
        while remaining:
            progressed = False
            next_rem: list[int] = []
            for ni in remaining:
                n_name, _, p_ni, _, _, world_mat = nodes_meta[ni]
                if p_ni >= 0 and nodes_meta[p_ni][1] in (0, 3) and p_ni not in node_objects:
                    next_rem.append(ni)
                    continue
                empty_obj = bpy.data.objects.new(name=n_name, object_data=None)
                collection.objects.link(empty_obj)
                _set_object_parent(empty_obj, node_objects.get(p_ni), world_mat, node_world_mats.get(p_ni))
                node_objects[ni] = empty_obj
                node_world_mats[ni] = world_mat
                created.append(empty_obj)
                progressed = True
            if not progressed:
                for ni in next_rem:
                    n_name, _, p_ni, _, _, world_mat = nodes_meta[ni]
                    empty_obj = bpy.data.objects.new(name=n_name, object_data=None)
                    collection.objects.link(empty_obj)
                    _set_object_parent(empty_obj, node_objects.get(p_ni), world_mat, node_world_mats.get(p_ni))
                    node_objects[ni] = empty_obj
                    node_world_mats[ni] = world_mat
                    created.append(empty_obj)
                break
            remaining = next_rem

        # Parent Armature under its FBX container Empty (if any) while preserving world Identity
        if arm_obj is not None and arm_node_idx >= 0:
            arm_parent_ni = nodes_meta[arm_node_idx][2]
            if arm_parent_ni in node_objects:
                _set_object_parent(
                    arm_obj,
                    node_objects[arm_parent_ni],
                    Matrix.Identity(4),
                    node_world_mats.get(arm_parent_ni),
                )

        # 5. Build Meshes
        for mi in range(n_meshes):
            mesh_obj = _build_mesh_object(
                lib, handle, mi, scene_mats, bone_names, arm_obj, node_objects, node_world_mats, nodes_meta, collection
            )
            created.append(mesh_obj)

        # Ensure Blender updates newly imported world matrices & object relations
        bpy.context.view_layer.update()
        return created
    finally:
        lib.scsp_fbx_free(handle)
