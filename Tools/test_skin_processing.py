"""Blender integration test for SCSP character skin adaptation and PMX toon export."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eye_texture_bake
import facial_morphs
import pmx_io
import preprocess_parts
import skin_processing
import studio_export


def _make_solid_png(path: Path, rgba: tuple[float, float, float, float]) -> None:
    pixels = np.empty((16, 16, 4), dtype=np.float32)
    pixels[..., :] = rgba
    owned: list[bpy.types.Image] = []
    try:
        eye_texture_bake.write_noncolor_rgba_png(path, pixels, owned)
    finally:
        for img in reversed(owned):
            bpy.data.images.remove(img)


def _new_material_with_image(name: str, image: bpy.types.Image) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _make_triangle_mesh(name: str, materials: list[bpy.types.Material]) -> bpy.types.Object:
    mesh_data = bpy.data.meshes.new(name)
    verts = []
    faces = []
    for idx in range(len(materials)):
        base = idx * 3
        verts.extend(
            [
                (float(idx), 0.0, 0.0),
                (float(idx) + 0.5, 0.0, 0.0),
                (float(idx), 0.5, 0.0),
            ]
        )
        faces.append((base, base + 1, base + 2))
    mesh_data.from_pydata(verts, [], faces)
    uv_layer = mesh_data.uv_layers.new(name="UVMap")
    for idx, poly in enumerate(mesh_data.polygons):
        poly.material_index = idx
        for loop_offset, uv in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))):
            uv_layer.data[poly.loop_start + loop_offset].uv = uv
    obj = bpy.data.objects.new(name, mesh_data)
    obj["scsp_merged_part"] = True
    bpy.context.collection.objects.link(obj)
    for mat in materials:
        obj.data.materials.append(mat)
    return obj


def _make_rig() -> bpy.types.Object:
    root = bpy.data.objects.new("SCSP_Character", None)
    bpy.context.collection.objects.link(root)
    arm_data = bpy.data.armatures.new("SCSP_Skeleton")
    rig = bpy.data.objects.new("SCSP_Skeleton", arm_data)
    bpy.context.collection.objects.link(rig)
    rig.parent = root
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bone = arm_data.edit_bones.new("全ての親")
    bone.head = (0.0, 0.0, 0.0)
    bone.tail = (0.0, 0.0, 0.2)
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def main() -> None:
    # Pure NumPy skin pixel & toon ramp math verification.
    shade = np.array([[[1.0, 0.0, 0.0, 1.0], [0.80, 0.0, 0.0, 1.0]]], dtype=np.float32)
    white = np.ones((1, 2, 4), dtype=np.float32)
    face_baked = skin_processing.bake_skin_pixels(
        white, "#693B3A", "#4B1214", shade_rgba=shade
    )
    np.testing.assert_allclose(
        face_baked[0, 0, :3], skin_processing.skin_tint("#693B3A"), atol=1e-5
    )
    np.testing.assert_allclose(
        face_baked[0, 1, :3],
        skin_processing.skin_tint("#693B3A", "#4B1214", 0.0),
        atol=1e-5,
    )
    ramp = skin_processing.make_toon_ramp_pixels("#663735", "#4A0C0D")
    assert ramp.shape == (skin_processing.TOON_HEIGHT, skin_processing.TOON_WIDTH, 4)
    np.testing.assert_allclose(ramp[-1, 0], [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    assert ramp[0, 0, 1] < ramp[0, 0, 0] < 0.95

    bpy.ops.wm.read_factory_settings(use_empty=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        face_col_path = root / "head_19" / "19_face_00_col.png"
        face_sd_path = root / "head_19" / "19_face_sd.png"
        body_col_path = root / "dress" / "00_bodyskin_00_col.png"
        cloth_col_path = root / "dress" / "dress_00_col.png"

        _make_solid_png(face_col_path, (230 / 255.0, 170 / 255.0, 160 / 255.0, 1.0))
        _make_solid_png(face_sd_path, (1.0, 0.0, 0.0, 1.0))
        _make_solid_png(body_col_path, (1.0, 239 / 255.0, 226 / 255.0, 0.5))
        _make_solid_png(cloth_col_path, (0.2, 0.4, 0.8, 1.0))

        face_shared_img = bpy.data.images.load(str(face_col_path), check_existing=False)
        body_shared_img = bpy.data.images.load(str(body_col_path), check_existing=False)
        cloth_img = bpy.data.images.load(str(cloth_col_path), check_existing=False)

        m_face = _new_material_with_image("M_face", face_shared_img)
        m_mouth = _new_material_with_image("M_mouth_inner", face_shared_img)
        m_teeth = _new_material_with_image("M_teeth_tongue", face_shared_img)
        m_skin = _new_material_with_image("M_0045_18NLRS_skin_00", body_shared_img)
        m_skin_bloom = _new_material_with_image(
            "M_0045_18NLRS_skin_00_bloom", body_shared_img
        )
        m_cloth = _new_material_with_image("M_cloth_00", cloth_img)

        rig = _make_rig()
        head_mesh = _make_triangle_mesh("face", [m_face, m_mouth, m_teeth])
        head_mesh.shape_key_add(name="Basis", from_mix=False)
        sk = head_mesh.shape_key_add(name="まばたき", from_mix=False)
        sk.data[0].co.y += 0.01
        body_mesh = _make_triangle_mesh("bodyskin", [m_skin, m_skin_bloom, m_cloth])
        for mesh in (head_mesh, body_mesh):
            mod = mesh.modifiers.new("Armature", "ARMATURE")
            mod.object = rig
            group = mesh.vertex_groups.new(name="全ての親")
            group.add(list(range(len(mesh.data.vertices))), 1.0, "REPLACE")

        skin_out = root / "skin-textures"
        parts = [
            {"kind": "head", "sourcepath": str(root / "head_19" / "M_Head_019.fbx"), "meshes": [head_mesh]},
            {"kind": "dress", "sourcepath": str(root / "dress" / "M_Dress.fbx"), "meshes": [body_mesh]},
        ]
        config = {
            "record": {
                "id": 19,
                "bodySkinBrightCode": "#663735",
                "bodySkinDarkCode": "#4A0C0D",
                "faceSkinBrightCode": "#693B3A",
                "faceSkinDarkCode": "#4B1214",
            },
            "skin_texture_output_dir": str(skin_out),
        }

        skin_report = skin_processing.prepare(parts, config)
        assert skin_report["status"] == "ok", skin_report
        assert not skin_report["warnings"], skin_report

        # Shared face_col datablock must remain intact on M_mouth_inner & M_teeth_tongue.
        assert m_mouth.node_tree.nodes["Image Texture"].image == face_shared_img
        assert m_teeth.node_tree.nodes["Image Texture"].image == face_shared_img
        face_node_img = m_face.node_tree.nodes["Image Texture"].image
        assert face_node_img != face_shared_img
        assert "face_19_face_00_col_skin" in Path(face_node_img.filepath).name

        # Both body skin materials must share one baked body skin image.
        skin_img_a = m_skin.node_tree.nodes["Image Texture"].image
        skin_img_b = m_skin_bloom.node_tree.nodes["Image Texture"].image
        assert skin_img_a == skin_img_b
        assert skin_img_a != body_shared_img

        # Verify baked body skin pixel matches Izumi Mei tint & preserves alpha.
        owned: list[bpy.types.Image] = []
        try:
            _, baked_body = eye_texture_bake.load_noncolor_pixels(
                Path(skin_img_a.filepath), owned
            )
            rgb_8bit = tuple(int(v) for v in np.rint(baked_body[0, 0, :3] * 255.0))
            assert rgb_8bit == (230, 171, 160), rgb_8bit
            assert abs(float(baked_body[0, 0, 3]) - 0.5) < 0.01
        finally:
            for img in reversed(owned):
                bpy.data.images.remove(img)

        # Verify duplicate_and_convert_materials -> prepare_textures -> PMX export preserves toon & material invariants.
        blend_path = root / "preprocessed.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        pmx_path = root / "export" / "test.pmx"
        pmx_report = studio_export.run(
            type(
                "Args",
                (),
                {
                    "input": str(blend_path),
                    "output": str(pmx_path),
                    "report": str(root / "export" / "report.json"),
                    "name": "MeiTest",
                    "comment": "Skin Test",
                },
            )(),
            {},
        )
        pmx_model = pmx_io.load(str(pmx_path))
        by_name = {mat.name: mat for mat in pmx_model.materials}
        assert len(by_name) == 6, by_name.keys()

        assert by_name["M_face"].is_shared_toon_texture is False
        assert by_name["M_face"].toon_texture >= 0
        assert by_name["M_0045_18NLRS_skin_00"].is_shared_toon_texture is False
        assert by_name["M_0045_18NLRS_skin_00"].toon_texture >= 0
        assert by_name["M_0045_18NLRS_skin_00_bloom"].is_shared_toon_texture is False
        assert (
            by_name["M_0045_18NLRS_skin_00"].toon_texture
            == by_name["M_0045_18NLRS_skin_00_bloom"].toon_texture
        )
        assert by_name["M_face"].toon_texture != by_name["M_0045_18NLRS_skin_00"].toon_texture
        assert by_name["M_mouth_inner"].toon_texture == -1
        assert by_name["M_teeth_tongue"].toon_texture == -1
        assert by_name["M_cloth_00"].toon_texture == -1

        for mat in (by_name["M_face"], by_name["M_0045_18NLRS_skin_00"]):
            tex_path = Path(pmx_model.textures[mat.texture].path)
            toon_path = Path(pmx_model.textures[mat.toon_texture].path)
            assert tex_path.is_file(), tex_path
            assert toon_path.is_file(), toon_path

        assert pmx_report["status"] == "ok"
        print("SKIN_PROCESSING_BLENDER_TEST_OK")


def test_accessory_face_makeup_and_helper_hiding() -> None:
    """Verify full-mesh & 4-vert makeup transfer + helper hiding for ATAMA_shadow/RoomShoes_BodySkin."""
    for vertex_count in (4, 15):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            head_dir = root / "head_20"
            acce_dir = root / "acce_92647_20anwd_00_20"
            dress_dir = root / "dress_0109"
            head_dir.mkdir(parents=True)
            acce_dir.mkdir(parents=True)
            dress_dir.mkdir(parents=True)

            base_col = head_dir / "20_face_00_col.png"
            makeup_col = acce_dir / "20_face_2647_col.png"
            dress_col = dress_dir / "0109_dress_00_col.png"
            _make_solid_png(base_col, (0.9, 0.8, 0.75, 1.0))
            _make_solid_png(makeup_col, (0.95, 0.55, 0.65, 1.0))
            _make_solid_png(dress_col, (0.2, 0.3, 0.4, 1.0))

            head_fbx = head_dir / "M_Head_20.fbx"
            acce_fbx = acce_dir / "M_Acce_92647_20ANWD_00_20.fbx"
            dress_fbx = dress_dir / "M_Dress_0109.fbx"
            for fbx in (head_fbx, acce_fbx, dress_fbx):
                fbx.write_bytes(b"")

            m_face = _new_material_with_image(
                "M_face", bpy.data.images.load(str(base_col), check_existing=False)
            )
            m_makeup = _new_material_with_image(
                "M_face_2647_20",
                bpy.data.images.load(str(makeup_col), check_existing=False),
            )
            m_lambert = bpy.data.materials.new(name="lambert1")
            m_dress = _new_material_with_image(
                "M_0109_dress_00",
                bpy.data.images.load(str(dress_col), check_existing=False),
            )

            face_obj = _make_triangle_mesh("face", [m_face])
            face_obj["scsp_source_object"] = "face"
            face_obj["scsp_source_fbx"] = str(head_fbx)

            # Build either a 4-vertex card or multi-vertex full makeup face.
            mesh_data = bpy.data.meshes.new("M_face_acce")
            verts = [(float(i), 0.0, 0.0) for i in range(vertex_count)]
            polys = [(0, 1, 2, 3)] if vertex_count == 4 else [(i, i + 1, i + 2) for i in range(0, vertex_count - 2, 3)]
            mesh_data.from_pydata(verts, [], polys)
            acce_face = bpy.data.objects.new("04_acce__M_face", mesh_data)
            bpy.context.collection.objects.link(acce_face)
            acce_face.data.materials.append(m_makeup)
            acce_face["scsp_source_object"] = "M_face"
            acce_face["scsp_source_fbx"] = str(acce_fbx)
            acce_face["scsp_merged_part"] = "acce"

            earring = _make_triangle_mesh("04_acce__earring", [m_dress])
            earring["scsp_source_object"] = "earring"
            earring["scsp_source_fbx"] = str(acce_fbx)

            atama_shadow = _make_triangle_mesh("ATAMA_shadow", [m_lambert])
            atama_shadow["scsp_source_object"] = "ATAMA_shadow"
            atama_shadow["scsp_source_fbx"] = str(head_fbx)

            m_eyeshadow = _new_material_with_image(
                "M_eyeshadow", bpy.data.images.load(str(dress_col), check_existing=False)
            )
            close_face = _make_triangle_mesh("closeface", [m_face])
            close_face["scsp_source_object"] = "closeface"
            close_face["scsp_source_fbx"] = str(head_fbx)

            eyeshadow = _make_triangle_mesh("eyeshadow", [m_eyeshadow])
            eyeshadow["scsp_source_object"] = "eyeshadow"
            eyeshadow["scsp_source_fbx"] = str(head_fbx)

            roomshoes_skin = _make_triangle_mesh("RoomShoes_BodySkin", [m_dress])
            roomshoes_skin["scsp_source_object"] = "RoomShoes_BodySkin"
            roomshoes_skin["scsp_source_fbx"] = str(dress_fbx)

            tri_trim = _make_triangle_mesh("triangle", [m_dress])
            tri_trim["scsp_source_object"] = "triangle"
            tri_trim["scsp_source_fbx"] = str(dress_fbx)

            parts = [
                {
                    "kind": "head",
                    "arm": None,
                    "objects": [face_obj, atama_shadow, close_face, eyeshadow],
                    "meshes": [face_obj, atama_shadow, close_face, eyeshadow],
                },
                {
                    "kind": "acce",
                    "arm": None,
                    "objects": [acce_face, earring],
                    "meshes": [acce_face, earring],
                },
                {
                    "kind": "dress",
                    "arm": None,
                    "objects": [roomshoes_skin, tri_trim],
                    "meshes": [roomshoes_skin, tri_trim],
                },
            ]

            report = preprocess_parts.prepare_parts(parts, eye_bindings=False)
            assert len(report["makeup_face_transfers"]) == 1, report["makeup_face_transfers"]
            assert Path(
                m_face.node_tree.nodes["Principled BSDF"]
                .inputs["Base Color"]
                .links[0]
                .from_node.image.filepath
            ).resolve() == makeup_col.resolve()

            hidden_reasons = {item["source_object"]: item["reason"] for item in report["hidden_objects"]}
            assert hidden_reasons == {
                "M_face": "accessory_face_texture_card",
                "ATAMA_shadow": "shadow_proxy",
                "closeface": "face_close",
                "RoomShoes_BodySkin": "room_shoes_helper",
            }, hidden_reasons
            assert acce_face.hide_render and atama_shadow.hide_render and close_face.hide_render and roomshoes_skin.hide_render
            assert not face_obj.hide_render and not earring.hide_render and not eyeshadow.hide_render and not tri_trim.hide_render
    print("ACCESSORY_MAKEUP_AND_HELPER_HIDING_OK")


def test_niyari_morph_symmetric_smile() -> None:
    """Verify generated `にやり` widens symmetrically and lifts both corners cleanly."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    m_face = bpy.data.materials.new(name="M_face")
    m_mouth = bpy.data.materials.new(name="M_mouthline")

    mesh_data = bpy.data.meshes.new("head")
    # 5 symmetric mouthline vertices (-0.015 .. +0.015) + 3 surrounding M_face vertices
    verts = [
        (-0.015, -0.095, 1.380),
        (-0.008, -0.098, 1.380),
        (0.000, -0.099, 1.380),
        (0.008, -0.098, 1.380),
        (0.015, -0.095, 1.380),
        (-0.025, -0.090, 1.385),
        (0.000, -0.099, 1.370),
        (0.025, -0.090, 1.385),
    ]
    polys = [(0, 1, 2), (2, 3, 4), (5, 0, 6), (6, 4, 7)]
    mesh_data.from_pydata(verts, [], polys)
    for i, poly in enumerate(mesh_data.polygons):
        poly.material_index = 1 if i < 2 else 0
    obj = bpy.data.objects.new("head", mesh_data)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(m_face)
    obj.data.materials.append(m_mouth)

    obj.shape_key_add(name="Basis", from_mix=False)
    majime = obj.shape_key_add(name="真面目口", from_mix=False)
    # Simulate a narrowed serious key to ensure `にやり` stays full-width on Basis.
    for i in range(5):
        majime.data[i].co.x *= 0.6

    resources = {
        "source_names": {"mouth_serious": "真面目口"},
        "generated_morphs": {"にやり": {"panel": 3, "english": "grin"}},
    }
    facial_morphs.generate([obj], resources)
    niyari = obj.data.shape_keys.key_blocks["にやり"]
    basis = obj.data.shape_keys.key_blocks["Basis"]

    left_dx = float(niyari.data[0].co.x - basis.data[0].co.x)
    right_dx = float(niyari.data[4].co.x - basis.data[4].co.x)
    left_dz = float(niyari.data[0].co.z - basis.data[0].co.z)
    right_dz = float(niyari.data[4].co.z - basis.data[4].co.z)

    assert left_dx < -0.001 and right_dx > 0.001, (left_dx, right_dx)
    assert abs(left_dx + right_dx) < 1e-5, (left_dx, right_dx)
    assert left_dz > 0.002 and right_dz > 0.002, (left_dz, right_dz)
    assert abs(left_dz - right_dz) < 1e-5, (left_dz, right_dz)
    print("NIYARI_SYMMETRIC_SMILE_OK")


def test_accessory_eyeparts_and_geometric_duplicate_auto_hiding() -> None:
    """Verify accessory eyeparts makeup transfer + geometric duplicate overlay hiding."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        head_dir = root / "head_17"
        acce_dir = root / "acce_92421_17desr_00_17"
        head_dir.mkdir(parents=True)
        acce_dir.mkdir(parents=True)

        base_face = head_dir / "17_face_00_col.png"
        base_eyeparts = head_dir / "17_eyeparts_00_col.png"
        makeup_eyeparts = acce_dir / "17_eyeparts_2421_col.png"
        for p in (base_face, base_eyeparts, makeup_eyeparts):
            img = bpy.data.images.new(p.name, width=2, height=2)
            img.filepath_raw = str(p)
            img.file_format = "PNG"
            img.save()

        head_fbx = head_dir / "M_Head_17.fbx"
        acce_fbx = acce_dir / "M_Acce_92421_17DESR_00_17.fbx"
        head_fbx.write_bytes(b"head")
        acce_fbx.write_bytes(b"acce")

        img_base_face = bpy.data.images.load(str(base_face))
        img_base_eyeparts = bpy.data.images.load(str(base_eyeparts))
        img_makeup_eyeparts = bpy.data.images.load(str(makeup_eyeparts))

        m_face = _new_material_with_image("M_face", img_base_face)
        m_eyelash = _new_material_with_image("M_eyelash", img_base_eyeparts)
        m_eyebrow = _new_material_with_image("M_eyebrow", img_base_eyeparts)
        m_acce_eye = _new_material_with_image("M_eyelash_2421_17", img_makeup_eyeparts)
        m_acce_unknown = bpy.data.materials.new(name="M_CustomOverlay")

        # 16+ vertices so geometric duplicate detector matches head face geometry cleanly.
        coords = [(float(i % 4) * 0.02, -0.08, 1.35 + float(i // 4) * 0.02) for i in range(16)]
        polys = [
            (r * 4 + c, r * 4 + c + 1, (r + 1) * 4 + c + 1, (r + 1) * 4 + c)
            for r in range(3)
            for c in range(3)
        ]

        def _make_grid(name, mat, src_obj, src_fbx):
            md = bpy.data.meshes.new(name)
            md.from_pydata(coords, [], polys)
            obj = bpy.data.objects.new(name, md)
            bpy.context.collection.objects.link(obj)
            obj.data.materials.append(mat)
            obj["scsp_source_object"] = src_obj
            obj["scsp_source_fbx"] = str(src_fbx)
            return obj

        head_face = _make_grid("00_head__face", m_face, "face", head_fbx)
        head_face.shape_key_add(name="Basis", from_mix=False)
        head_lash = _make_triangle_mesh("00_head__eyelash_L", [m_eyelash])
        head_lash["scsp_source_object"] = "eyelash_L"
        head_lash["scsp_source_fbx"] = str(head_fbx)
        head_brow = _make_triangle_mesh("00_head__eyebrow_L", [m_eyebrow])
        head_brow["scsp_source_object"] = "eyebrow_L"
        head_brow["scsp_source_fbx"] = str(head_fbx)

        acce_eyeparts_card = _make_grid("05_acce__M_eyelash", m_acce_eye, "M_eyelash", acce_fbx)
        acce_unknown_dup = _make_grid("05_acce__WeirdDuplicateFace", m_acce_unknown, "WeirdDuplicateFace", acce_fbx)
        real_pin = _make_triangle_mesh("05_acce__face_metal", [m_acce_unknown])
        real_pin["scsp_source_object"] = "face_metal"
        real_pin["scsp_source_fbx"] = str(acce_fbx)

        parts = [
            {
                "kind": "head",
                "arm": None,
                "objects": [head_face, head_lash, head_brow],
                "meshes": [head_face, head_lash, head_brow],
            },
            {
                "kind": "acce",
                "arm": None,
                "objects": [acce_eyeparts_card, acce_unknown_dup, real_pin],
                "meshes": [acce_eyeparts_card, acce_unknown_dup, real_pin],
            },
        ]

        report = preprocess_parts.prepare_parts(parts, eye_bindings=False)
        hidden_reasons = {item["source_object"]: item["reason"] for item in report["hidden_objects"]}
        assert hidden_reasons == {
            "M_eyelash": "accessory_eyeparts_texture_card",
            "WeirdDuplicateFace": "accessory_head_duplicate_overlay",
        }, hidden_reasons
        assert acce_eyeparts_card.hide_render and acce_unknown_dup.hide_render
        assert not real_pin.hide_render
        assert Path(
            m_eyelash.node_tree.nodes["Principled BSDF"]
            .inputs["Base Color"]
            .links[0]
            .from_node.image.filepath
        ).resolve() == makeup_eyeparts.resolve()
        assert Path(
            m_eyebrow.node_tree.nodes["Principled BSDF"]
            .inputs["Base Color"]
            .links[0]
            .from_node.image.filepath
        ).resolve() == makeup_eyeparts.resolve()
    print("ACCESSORY_EYEPARTS_AND_GEOM_DUPLICATE_AUTO_OK")


if __name__ == "__main__":
    main()
    test_accessory_face_makeup_and_helper_hiding()
    test_niyari_morph_symmetric_smile()
    test_accessory_eyeparts_and_geometric_duplicate_auto_hiding()
