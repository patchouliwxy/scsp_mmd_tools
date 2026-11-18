"""Clean-room pure-Python PMX 2.0 / 2.1 binary reader and writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Any, BinaryIO


PMX_MAGIC = b"PMX "

_VERTEX_INDEX_FMT = {1: "<B", 2: "<H", 4: "<i"}
_SIGNED_INDEX_FMT = {1: "<b", 2: "<h", 4: "<i"}


@dataclass
class Header:
    sign: bytes = PMX_MAGIC
    version: float = 2.0
    encoding: str = "utf-16-le"
    additional_uvs: int = 0
    vertex_index_size: int = 4
    texture_index_size: int = 1
    material_index_size: int = 1
    bone_index_size: int = 2
    morph_index_size: int = 2
    rigid_index_size: int = 2


@dataclass
class SDEFWeight:
    weight: float = 0.0
    c: tuple[float, float, float] = (0.0, 0.0, 0.0)
    r0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    r1: tuple[float, float, float] = (0.0, 0.0, 0.0)


class BoneWeight:
    BDEF1 = 0
    BDEF2 = 1
    BDEF4 = 2
    SDEF = 3
    QDEF = 4

    def __init__(self) -> None:
        self.type: int = self.BDEF1
        self.bones: list[int] = [0]
        self.weights: Any = []


class Vertex:
    def __init__(self) -> None:
        self.co: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.normal: list[float] | tuple[float, float, float] = [0.0, 1.0, 0.0]
        self.uv: list[float] | tuple[float, float] = [0.0, 0.0]
        self.additional_uvs: list[Any] = []
        self.weight: BoneWeight = BoneWeight()
        self.edge_scale: float = 1.0


@dataclass
class Texture:
    path: str = ""


class Material:
    def __init__(self) -> None:
        self.name: str = ""
        self.name_e: str = ""
        self.diffuse: list[float] | tuple[float, float, float, float] = [1.0, 1.0, 1.0, 1.0]
        self.specular: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.shininess: float = 1.0
        self.ambient: list[float] | tuple[float, float, float] = [0.25, 0.25, 0.25]
        self.is_double_sided: bool = True
        self.enabled_drop_shadow: bool = True
        self.enabled_self_shadow_map: bool = True
        self.enabled_self_shadow: bool = True
        self.enabled_toon_edge: bool = False
        self.enabled_vertex_color: bool = False
        self.enabled_point_draw: bool = False
        self.enabled_line_draw: bool = False
        self.edge_color: list[float] | tuple[float, float, float, float] = [0.0, 0.0, 0.0, 1.0]
        self.edge_size: float = 1.0
        self.texture: int = -1
        self.sphere_texture: int = -1
        self.sphere_texture_mode: int = 0
        self.is_shared_toon_texture: bool = True
        self.toon_texture: int = 0
        self.comment: str = ""
        self.vertex_count: int = 0


@dataclass
class Coordinate:
    x_axis: list[float] | tuple[float, float, float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    z_axis: list[float] | tuple[float, float, float] = field(default_factory=lambda: [0.0, 0.0, 1.0])


class IKLink:
    def __init__(self) -> None:
        self.target: int = 0
        self.minimumAngle: list[float] | tuple[float, float, float] | None = None
        self.maximumAngle: list[float] | tuple[float, float, float] | None = None


class Bone:
    def __init__(self) -> None:
        self.name: str = ""
        self.name_e: str = ""
        self.location: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.parent: int = -1
        self.transform_order: int = 0
        self.displayConnection: int | list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.isRotatable: bool = True
        self.isMovable: bool = False
        self.visible: bool = True
        self.isControllable: bool = True
        self.isIK: bool = False
        self.isLocalAdditional: bool = False
        self.hasAdditionalRotate: bool = False
        self.hasAdditionalLocation: bool = False
        self.additionalTransform: tuple[int, float] | None = None
        self.axis: list[float] | tuple[float, float, float] | None = None
        self.localCoordinate: Coordinate | None = None
        self.transAfterPhis: bool = False
        self.externalTransKey: int | None = None
        self.target: int | None = None
        self.loopCount: int = 8
        self.rotationConstraint: float = 0.03
        self.ik_links: list[IKLink] = []


@dataclass
class MorphData:
    morph: int = 0
    factor: float = 1.0


@dataclass
class VertexMorphOffset:
    index: int = 0
    offset: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BoneMorphOffset:
    index: int = 0
    location_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class UVMorphOffset:
    index: int = 0
    offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class MaterialMorphData:
    index: int = -1
    offset_type: int = 0
    diffuse: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shininess: float = 0.0
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    edge_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    edge_size: float = 0.0
    texture_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    sphere_texture_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    toon_texture_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass
class ImpulseMorphData:
    rigid_index: int = 0
    is_local: int = 0
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0)


class _BaseMorph:
    morph_type: int = -1

    def __init__(self, name: str = "", name_e: str = "", category: int = 4) -> None:
        self.name: str = name
        self.name_e: str = name_e
        self.category: int = category
        self.offsets: list[Any] = []


class GroupMorph(_BaseMorph):
    morph_type = 0


class VertexMorph(_BaseMorph):
    morph_type = 1


class BoneMorph(_BaseMorph):
    morph_type = 2


class UVMorph(_BaseMorph):
    def __init__(self, name: str = "", name_e: str = "", category: int = 4, uv_index: int = 0) -> None:
        super().__init__(name=name, name_e=name_e, category=category)
        self.uv_index = uv_index
        self.morph_type = 3 + uv_index


class MaterialMorph(_BaseMorph):
    morph_type = 8


class FlipMorph(_BaseMorph):
    morph_type = 9


class ImpulseMorph(_BaseMorph):
    morph_type = 10


class Display:
    def __init__(self) -> None:
        self.name: str = ""
        self.name_e: str = ""
        self.isSpecial: bool = False
        self.data: list[tuple[int, int]] = []


class Rigid:
    MODE_STATIC = 0
    MODE_DYNAMIC = 1
    MODE_DYNAMIC_AND_BONE = 2
    MODE_DYNAMIC_BONE = 2

    TYPE_SPHERE = 0
    TYPE_BOX = 1
    TYPE_CAPSULE = 2

    def __init__(self) -> None:
        self.name: str = ""
        self.name_e: str = ""
        self.bone: int = -1
        self.collision_group_number: int = 0
        self.collision_group_mask: int = 0xFFFF
        self.type: int = self.TYPE_SPHERE
        self.size: list[float] | tuple[float, float, float] = [1.0, 1.0, 1.0]
        self.location: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.rotation: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.mass: float = 1.0
        self.velocity_attenuation: float = 0.5
        self.rotation_attenuation: float = 0.5
        self.bounce: float = 0.0
        self.friction: float = 0.5
        self.mode: int = self.MODE_STATIC


class Joint:
    MODE_SPRING6DOF = 0

    def __init__(self) -> None:
        self.name: str = ""
        self.name_e: str = ""
        self.mode: int = self.MODE_SPRING6DOF
        self.src_rigid: int = -1
        self.dest_rigid: int = -1
        self.location: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.rotation: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.minimum_location: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.maximum_location: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.minimum_rotation: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.maximum_rotation: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.spring_constant: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]
        self.spring_rotation_constant: list[float] | tuple[float, float, float] = [0.0, 0.0, 0.0]


class Model:
    def __init__(self) -> None:
        self.header: Header = Header()
        self.name: str = ""
        self.name_e: str = ""
        self.comment: str = ""
        self.comment_e: str = ""
        self.vertices: list[Vertex] = []
        self.faces: list[tuple[int, int, int]] = []
        self.textures: list[Texture] = []
        self.materials: list[Material] = []
        self.bones: list[Bone] = []
        self.morphs: list[Any] = []
        self.display: list[Display] = []
        self.rigids: list[Rigid] = []
        self.joints: list[Joint] = []


class _Reader:
    def __init__(self, stream: BinaryIO) -> None:
        self._buf = stream.read()
        self._pos = 0

    def read(self, n: int) -> bytes:
        if self._pos + n > len(self._buf):
            raise EOFError(f"Unexpected EOF at {self._pos} (wanted {n} bytes)")
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        s = struct.Struct(fmt)
        if self._pos + s.size > len(self._buf):
            raise EOFError(f"Unexpected EOF at {self._pos} (wanted {s.size} bytes for {fmt})")
        val = s.unpack_from(self._buf, self._pos)
        self._pos += s.size
        return val

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u16(self) -> int:
        return self.unpack("<H")[0]

    def i32(self) -> int:
        return self.unpack("<i")[0]

    def f32(self) -> float:
        return self.unpack("<f")[0]

    def vec2(self) -> tuple[float, float]:
        return self.unpack("<2f")

    def vec3(self) -> tuple[float, float, float]:
        return self.unpack("<3f")

    def vec4(self) -> tuple[float, float, float, float]:
        return self.unpack("<4f")

    def text(self, encoding: str) -> str:
        length = self.i32()
        if length <= 0:
            return ""
        raw = self.read(length)
        return raw.decode(encoding)

    def idx(self, fmt: str) -> int:
        return self.unpack(fmt)[0]


def _vertex_index_size(count: int) -> int:
    if count <= 0xFF:
        return 1
    if count <= 0xFFFF:
        return 2
    return 4


def _signed_index_size(count: int) -> int:
    if count < 128:
        return 1
    if count < 32768:
        return 2
    return 4


def _pack_text(value: str, encoding: str) -> bytes:
    raw = (value or "").encode(encoding)
    return struct.pack("<i", len(raw)) + raw


def load(filepath: str | Path) -> Model:
    path = Path(filepath)
    with path.open("rb") as f:
        r = _Reader(f)

    sign = r.read(4)
    if sign != PMX_MAGIC:
        raise ValueError(f"Invalid PMX magic {sign!r} in {path}")
    version = r.f32()
    globals_len = r.u8()
    globals_bytes = list(r.read(globals_len))
    if len(globals_bytes) < 8:
        raise ValueError(f"PMX globals too short: {len(globals_bytes)}")

    encoding = "utf-16-le" if globals_bytes[0] == 0 else "utf-8"
    add_uv_count = globals_bytes[1]
    v_fmt = _VERTEX_INDEX_FMT[globals_bytes[2]]
    t_fmt = _SIGNED_INDEX_FMT[globals_bytes[3]]
    m_fmt = _SIGNED_INDEX_FMT[globals_bytes[4]]
    b_fmt = _SIGNED_INDEX_FMT[globals_bytes[5]]
    mo_fmt = _SIGNED_INDEX_FMT[globals_bytes[6]]
    rb_fmt = _SIGNED_INDEX_FMT[globals_bytes[7]]

    model = Model()
    model.header = Header(
        sign=sign,
        version=version,
        encoding=encoding,
        additional_uvs=add_uv_count,
        vertex_index_size=globals_bytes[2],
        texture_index_size=globals_bytes[3],
        material_index_size=globals_bytes[4],
        bone_index_size=globals_bytes[5],
        morph_index_size=globals_bytes[6],
        rigid_index_size=globals_bytes[7],
    )

    model.name = r.text(encoding)
    model.name_e = r.text(encoding)
    model.comment = r.text(encoding)
    model.comment_e = r.text(encoding)

    # Vertices
    v_count = r.i32()
    vertices: list[Vertex] = []
    for _ in range(v_count):
        v = Vertex()
        v.co = r.vec3()
        v.normal = r.vec3()
        v.uv = r.vec2()
        v.additional_uvs = [r.vec4() for _ in range(add_uv_count)]
        w_type = r.u8()
        bw = BoneWeight()
        bw.type = w_type
        if w_type == BoneWeight.BDEF1:
            bw.bones = [r.idx(b_fmt)]
            bw.weights = []
        elif w_type == BoneWeight.BDEF2:
            b0 = r.idx(b_fmt)
            b1 = r.idx(b_fmt)
            w0 = r.f32()
            bw.bones = [b0, b1]
            bw.weights = [w0]
        elif w_type in (BoneWeight.BDEF4, BoneWeight.QDEF):
            b0 = r.idx(b_fmt)
            b1 = r.idx(b_fmt)
            b2 = r.idx(b_fmt)
            b3 = r.idx(b_fmt)
            bw.bones = [b0, b1, b2, b3]
            bw.weights = r.vec4()
        elif w_type == BoneWeight.SDEF:
            b0 = r.idx(b_fmt)
            b1 = r.idx(b_fmt)
            w0 = r.f32()
            c = r.vec3()
            r0 = r.vec3()
            r1 = r.vec3()
            bw.bones = [b0, b1]
            bw.weights = SDEFWeight(weight=w0, c=c, r0=r0, r1=r1)
        else:
            raise ValueError(f"Unsupported PMX vertex weight type: {w_type}")
        v.weight = bw
        v.edge_scale = r.f32()
        vertices.append(v)
    model.vertices = vertices

    # Faces
    index_count = r.i32()
    if index_count % 3 != 0:
        raise ValueError(f"PMX face index count {index_count} is not divisible by 3")
    face_struct = struct.Struct("<" + v_fmt[1] * 3)
    faces: list[tuple[int, int, int]] = []
    for _ in range(index_count // 3):
        faces.append(face_struct.unpack_from(r._buf, r._pos))
        r._pos += face_struct.size
    model.faces = faces

    # Textures
    tex_count = r.i32()
    model.textures = [Texture(path=r.text(encoding)) for _ in range(tex_count)]

    # Materials
    mat_count = r.i32()
    materials: list[Material] = []
    for _ in range(mat_count):
        mat = Material()
        mat.name = r.text(encoding)
        mat.name_e = r.text(encoding)
        mat.diffuse = r.vec4()
        mat.specular = r.vec3()
        mat.shininess = r.f32()
        mat.ambient = r.vec3()
        flag = r.u8()
        mat.is_double_sided = bool(flag & 0x01)
        mat.enabled_drop_shadow = bool(flag & 0x02)
        mat.enabled_self_shadow_map = bool(flag & 0x04)
        mat.enabled_self_shadow = bool(flag & 0x08)
        mat.enabled_toon_edge = bool(flag & 0x10)
        mat.enabled_vertex_color = bool(flag & 0x20)
        mat.enabled_point_draw = bool(flag & 0x40)
        mat.enabled_line_draw = bool(flag & 0x80)
        mat.edge_color = r.vec4()
        mat.edge_size = r.f32()
        mat.texture = r.idx(t_fmt)
        mat.sphere_texture = r.idx(t_fmt)
        mat.sphere_texture_mode = r.u8()
        mat.is_shared_toon_texture = bool(r.u8())
        if mat.is_shared_toon_texture:
            mat.toon_texture = r.u8()
        else:
            mat.toon_texture = r.idx(t_fmt)
        mat.comment = r.text(encoding)
        mat.vertex_count = r.i32()
        materials.append(mat)
    model.materials = materials

    # Bones
    bone_count = r.i32()
    bones: list[Bone] = []
    for _ in range(bone_count):
        b = Bone()
        b.name = r.text(encoding)
        b.name_e = r.text(encoding)
        b.location = r.vec3()
        b.parent = r.idx(b_fmt)
        b.transform_order = r.i32()
        flag = r.u16()
        if flag & 0x0001:
            b.displayConnection = r.idx(b_fmt)
        else:
            b.displayConnection = r.vec3()
        b.isRotatable = bool(flag & 0x0002)
        b.isMovable = bool(flag & 0x0004)
        b.visible = bool(flag & 0x0008)
        b.isControllable = bool(flag & 0x0010)
        b.isIK = bool(flag & 0x0020)
        b.isLocalAdditional = bool(flag & 0x0080)
        b.hasAdditionalRotate = bool(flag & 0x0100)
        b.hasAdditionalLocation = bool(flag & 0x0200)
        b.transAfterPhis = bool(flag & 0x1000)

        if b.hasAdditionalRotate or b.hasAdditionalLocation:
            b.additionalTransform = (r.idx(b_fmt), r.f32())
        if flag & 0x0400:
            b.axis = r.vec3()
        if flag & 0x0800:
            b.localCoordinate = Coordinate(x_axis=r.vec3(), z_axis=r.vec3())
        if flag & 0x2000:
            b.externalTransKey = r.i32()
        if b.isIK:
            b.target = r.idx(b_fmt)
            b.loopCount = r.i32()
            b.rotationConstraint = r.f32()
            link_count = r.i32()
            links: list[IKLink] = []
            for _ in range(link_count):
                link = IKLink()
                link.target = r.idx(b_fmt)
                if r.u8():
                    link.minimumAngle = r.vec3()
                    link.maximumAngle = r.vec3()
                links.append(link)
            b.ik_links = links
        bones.append(b)
    model.bones = bones

    # Morphs
    morph_count = r.i32()
    morphs: list[Any] = []
    for _ in range(morph_count):
        m_name = r.text(encoding)
        m_name_e = r.text(encoding)
        cat = r.u8()
        m_type = r.u8()
        off_count = r.i32()
        if m_type == 0:
            mo: Any = GroupMorph(name=m_name, name_e=m_name_e, category=cat)
            mo.offsets = [MorphData(morph=r.idx(mo_fmt), factor=r.f32()) for _ in range(off_count)]
        elif m_type == 1:
            mo = VertexMorph(name=m_name, name_e=m_name_e, category=cat)
            mo.offsets = [VertexMorphOffset(index=r.idx(v_fmt), offset=r.vec3()) for _ in range(off_count)]
        elif m_type == 2:
            mo = BoneMorph(name=m_name, name_e=m_name_e, category=cat)
            mo.offsets = [
                BoneMorphOffset(index=r.idx(b_fmt), location_offset=r.vec3(), rotation_offset=r.vec4())
                for _ in range(off_count)
            ]
        elif 3 <= m_type <= 7:
            mo = UVMorph(name=m_name, name_e=m_name_e, category=cat, uv_index=m_type - 3)
            mo.offsets = [UVMorphOffset(index=r.idx(v_fmt), offset=r.vec4()) for _ in range(off_count)]
        elif m_type == 8:
            mo = MaterialMorph(name=m_name, name_e=m_name_e, category=cat)
            mo.offsets = [
                MaterialMorphData(
                    index=r.idx(m_fmt),
                    offset_type=r.u8(),
                    diffuse=r.vec4(),
                    specular=r.vec3(),
                    shininess=r.f32(),
                    ambient=r.vec3(),
                    edge_color=r.vec4(),
                    edge_size=r.f32(),
                    texture_factor=r.vec4(),
                    sphere_texture_factor=r.vec4(),
                    toon_texture_factor=r.vec4(),
                )
                for _ in range(off_count)
            ]
        elif m_type == 9:
            mo = FlipMorph(name=m_name, name_e=m_name_e, category=cat)
            mo.offsets = [MorphData(morph=r.idx(mo_fmt), factor=r.f32()) for _ in range(off_count)]
        elif m_type == 10:
            mo = ImpulseMorph(name=m_name, name_e=m_name_e, category=cat)
            mo.offsets = [
                ImpulseMorphData(
                    rigid_index=r.idx(rb_fmt),
                    is_local=r.u8(),
                    velocity=r.vec3(),
                    torque=r.vec3(),
                )
                for _ in range(off_count)
            ]
        else:
            raise ValueError(f"Unsupported PMX morph type: {m_type}")
        morphs.append(mo)
    model.morphs = morphs

    # Display frames
    disp_count = r.i32()
    displays: list[Display] = []
    for _ in range(disp_count):
        d = Display()
        d.name = r.text(encoding)
        d.name_e = r.text(encoding)
        d.isSpecial = bool(r.u8())
        item_count = r.i32()
        items: list[tuple[int, int]] = []
        for _ in range(item_count):
            t = r.u8()
            idx = r.idx(mo_fmt if t == 1 else b_fmt)
            items.append((t, idx))
        d.data = items
        displays.append(d)
    model.display = displays

    # Rigid bodies
    rigid_count = r.i32()
    rigids: list[Rigid] = []
    for _ in range(rigid_count):
        rb = Rigid()
        rb.name = r.text(encoding)
        rb.name_e = r.text(encoding)
        rb.bone = r.idx(b_fmt)
        rb.collision_group_number = r.u8()
        rb.collision_group_mask = r.u16()
        rb.type = r.u8()
        rb.size = r.vec3()
        rb.location = r.vec3()
        rb.rotation = r.vec3()
        rb.mass = r.f32()
        rb.velocity_attenuation = r.f32()
        rb.rotation_attenuation = r.f32()
        rb.bounce = r.f32()
        rb.friction = r.f32()
        rb.mode = r.u8()
        rigids.append(rb)
    model.rigids = rigids

    # Joints
    joint_count = r.i32()
    joints: list[Joint] = []
    for _ in range(joint_count):
        j = Joint()
        j.name = r.text(encoding)
        j.name_e = r.text(encoding)
        j.mode = r.u8()
        j.src_rigid = r.idx(rb_fmt)
        j.dest_rigid = r.idx(rb_fmt)
        j.location = r.vec3()
        j.rotation = r.vec3()
        j.minimum_location = r.vec3()
        j.maximum_location = r.vec3()
        j.minimum_rotation = r.vec3()
        j.maximum_rotation = r.vec3()
        j.spring_constant = r.vec3()
        j.spring_rotation_constant = r.vec3()
        joints.append(j)
    model.joints = joints

    return model


def save(filepath: str | Path, model: Model, add_uv_count: int = 0) -> None:
    path = Path(filepath)
    encoding = "utf-16-le"
    enc_byte = 0
    if model.header and model.header.encoding == "utf-8":
        encoding = "utf-8"
        enc_byte = 1

    if add_uv_count == 0 and model.vertices:
        add_uv_count = max(len(v.additional_uvs) for v in model.vertices)

    v_size = _vertex_index_size(len(model.vertices))
    t_size = _signed_index_size(len(model.textures))
    m_size = _signed_index_size(len(model.materials))
    b_size = _signed_index_size(len(model.bones))
    mo_size = _signed_index_size(len(model.morphs))
    rb_size = _signed_index_size(len(model.rigids))

    v_fmt = _VERTEX_INDEX_FMT[v_size]
    t_fmt = _SIGNED_INDEX_FMT[t_size]
    m_fmt = _SIGNED_INDEX_FMT[m_size]
    b_fmt = _SIGNED_INDEX_FMT[b_size]
    mo_fmt = _SIGNED_INDEX_FMT[mo_size]
    rb_fmt = _SIGNED_INDEX_FMT[rb_size]

    version = float(getattr(model.header, "version", 2.0) or 2.0)

    chunks: list[bytes] = []
    emit = chunks.append

    # Header & globals
    emit(PMX_MAGIC)
    emit(struct.pack("<fB", version, 8))
    emit(bytes([enc_byte, add_uv_count, v_size, t_size, m_size, b_size, mo_size, rb_size]))
    emit(_pack_text(model.name, encoding))
    emit(_pack_text(model.name_e, encoding))
    emit(_pack_text(model.comment, encoding))
    emit(_pack_text(model.comment_e, encoding))

    # Vertices
    emit(struct.pack("<i", len(model.vertices)))
    base_vert_fmt = f"<3f3f2f{add_uv_count * 4}fB" if add_uv_count > 0 else "<3f3f2fB"
    base_vert_struct = struct.Struct(base_vert_fmt)
    bdef1_struct = struct.Struct(b_fmt)
    bdef2_struct = struct.Struct("<" + b_fmt[1] * 2 + "f")
    bdef4_struct = struct.Struct("<" + b_fmt[1] * 4 + "4f")
    sdef_struct = struct.Struct("<" + b_fmt[1] * 2 + "f9f")
    f32_struct = struct.Struct("<f")

    for v in model.vertices:
        uv_vals: list[float] = []
        if add_uv_count > 0:
            for k in range(add_uv_count):
                if k < len(v.additional_uvs):
                    uv4 = v.additional_uvs[k]
                    uv_vals.extend((float(uv4[0]), float(uv4[1]), float(uv4[2]), float(uv4[3])))
                else:
                    uv_vals.extend((0.0, 0.0, 0.0, 0.0))
        bw = v.weight
        w_type = int(bw.type)
        emit(
            base_vert_struct.pack(
                float(v.co[0]),
                float(v.co[1]),
                float(v.co[2]),
                float(v.normal[0]),
                float(v.normal[1]),
                float(v.normal[2]),
                float(v.uv[0]),
                float(v.uv[1]),
                *uv_vals,
                w_type,
            )
        )
        if w_type == BoneWeight.BDEF1:
            emit(bdef1_struct.pack(int(bw.bones[0]) if bw.bones else 0))
        elif w_type == BoneWeight.BDEF2:
            w0 = float(bw.weights[0]) if bw.weights else 0.5
            emit(bdef2_struct.pack(int(bw.bones[0]), int(bw.bones[1]), w0))
        elif w_type in (BoneWeight.BDEF4, BoneWeight.QDEF):
            emit(
                bdef4_struct.pack(
                    int(bw.bones[0]),
                    int(bw.bones[1]),
                    int(bw.bones[2]),
                    int(bw.bones[3]),
                    float(bw.weights[0]),
                    float(bw.weights[1]),
                    float(bw.weights[2]),
                    float(bw.weights[3]),
                )
            )
        elif w_type == BoneWeight.SDEF:
            sw = bw.weights
            emit(
                sdef_struct.pack(
                    int(bw.bones[0]),
                    int(bw.bones[1]),
                    float(sw.weight),
                    float(sw.c[0]),
                    float(sw.c[1]),
                    float(sw.c[2]),
                    float(sw.r0[0]),
                    float(sw.r0[1]),
                    float(sw.r0[2]),
                    float(sw.r1[0]),
                    float(sw.r1[1]),
                    float(sw.r1[2]),
                )
            )
        else:
            raise ValueError(f"Unsupported PMX vertex weight type on save: {w_type}")
        emit(f32_struct.pack(float(v.edge_scale)))

    # Faces
    emit(struct.pack("<i", len(model.faces) * 3))
    face_struct = struct.Struct("<" + v_fmt[1] * 3)
    for f0, f1, f2 in model.faces:
        emit(face_struct.pack(int(f0), int(f1), int(f2)))

    # Textures
    emit(struct.pack("<i", len(model.textures)))
    for t in model.textures:
        emit(_pack_text(t.path, encoding))

    # Materials
    emit(struct.pack("<i", len(model.materials)))
    mat_tail_struct = struct.Struct("<4ff" + t_fmt[1] * 2 + "BB")
    for mat in model.materials:
        emit(_pack_text(mat.name, encoding))
        emit(_pack_text(mat.name_e, encoding))
        flag = (
            (0x01 if mat.is_double_sided else 0)
            | (0x02 if mat.enabled_drop_shadow else 0)
            | (0x04 if mat.enabled_self_shadow_map else 0)
            | (0x08 if mat.enabled_self_shadow else 0)
            | (0x10 if mat.enabled_toon_edge else 0)
            | (0x20 if getattr(mat, "enabled_vertex_color", False) else 0)
            | (0x40 if getattr(mat, "enabled_point_draw", False) else 0)
            | (0x80 if getattr(mat, "enabled_line_draw", False) else 0)
        )
        emit(
            struct.pack(
                "<4f3ff3fB",
                *(float(x) for x in mat.diffuse),
                *(float(x) for x in mat.specular),
                float(mat.shininess),
                *(float(x) for x in mat.ambient),
                flag,
            )
        )
        tex_idx = -1 if mat.texture is None else int(mat.texture)
        sph_idx = -1 if mat.sphere_texture is None else int(mat.sphere_texture)
        shared = bool(mat.is_shared_toon_texture)
        emit(
            mat_tail_struct.pack(
                *(float(x) for x in mat.edge_color),
                float(mat.edge_size),
                tex_idx,
                sph_idx,
                int(mat.sphere_texture_mode),
                1 if shared else 0,
            )
        )
        toon_idx = 0 if mat.toon_texture is None else int(mat.toon_texture)
        if shared:
            emit(struct.pack("<B", toon_idx & 0xFF))
        else:
            emit(struct.pack(t_fmt, -1 if mat.toon_texture is None else int(mat.toon_texture)))
        emit(_pack_text(mat.comment, encoding))
        emit(struct.pack("<i", int(mat.vertex_count)))

    # Bones
    emit(struct.pack("<i", len(model.bones)))
    for b in model.bones:
        emit(_pack_text(b.name, encoding))
        emit(_pack_text(b.name_e, encoding))
        parent = -1 if b.parent is None else int(b.parent)
        is_conn_bone = isinstance(b.displayConnection, int)
        has_add_rot = bool(b.hasAdditionalRotate)
        has_add_loc = bool(b.hasAdditionalLocation)
        has_axis = b.axis is not None
        has_local = b.localCoordinate is not None
        has_ext = b.externalTransKey is not None
        flag = (
            (0x0001 if is_conn_bone else 0)
            | (0x0002 if b.isRotatable else 0)
            | (0x0004 if b.isMovable else 0)
            | (0x0008 if b.visible else 0)
            | (0x0010 if b.isControllable else 0)
            | (0x0020 if b.isIK else 0)
            | (0x0080 if getattr(b, "isLocalAdditional", False) else 0)
            | (0x0100 if has_add_rot else 0)
            | (0x0200 if has_add_loc else 0)
            | (0x0400 if has_axis else 0)
            | (0x0800 if has_local else 0)
            | (0x1000 if b.transAfterPhis else 0)
            | (0x2000 if has_ext else 0)
        )
        emit(
            struct.pack(
                "<3f" + b_fmt[1] + "iH",
                float(b.location[0]),
                float(b.location[1]),
                float(b.location[2]),
                parent,
                int(b.transform_order),
                flag,
            )
        )
        if is_conn_bone:
            emit(struct.pack(b_fmt, int(b.displayConnection)))
        else:
            conn = b.displayConnection if b.displayConnection is not None else (0.0, 0.0, 0.0)
            emit(struct.pack("<3f", float(conn[0]), float(conn[1]), float(conn[2])))
        if has_add_rot or has_add_loc:
            t_idx, factor = b.additionalTransform if b.additionalTransform is not None else (-1, 0.0)
            emit(struct.pack("<" + b_fmt[1] + "f", int(t_idx), float(factor)))
        if has_axis:
            emit(struct.pack("<3f", *(float(x) for x in b.axis)))
        if has_local:
            emit(
                struct.pack(
                    "<6f",
                    *(float(x) for x in b.localCoordinate.x_axis),
                    *(float(x) for x in b.localCoordinate.z_axis),
                )
            )
        if has_ext:
            emit(struct.pack("<i", int(b.externalTransKey)))
        if b.isIK:
            ik_target = -1 if b.target is None else int(b.target)
            emit(
                struct.pack(
                    "<" + b_fmt[1] + "ifi",
                    ik_target,
                    int(b.loopCount),
                    float(b.rotationConstraint),
                    len(b.ik_links),
                )
            )
            for link in b.ik_links:
                has_limit = link.minimumAngle is not None and link.maximumAngle is not None
                emit(struct.pack("<" + b_fmt[1] + "B", int(link.target), 1 if has_limit else 0))
                if has_limit:
                    emit(
                        struct.pack(
                            "<6f",
                            *(float(x) for x in link.minimumAngle),
                            *(float(x) for x in link.maximumAngle),
                        )
                    )

    # Morphs
    emit(struct.pack("<i", len(model.morphs)))
    vm_struct = struct.Struct("<" + v_fmt[1] + "3f")
    gm_struct = struct.Struct("<" + mo_fmt[1] + "f")
    bm_struct = struct.Struct("<" + b_fmt[1] + "7f")
    uvm_struct = struct.Struct("<" + v_fmt[1] + "4f")
    mm_struct = struct.Struct("<" + m_fmt[1] + "B4f3ff3f4ff4f4f4f")
    imp_struct = struct.Struct("<" + rb_fmt[1] + "B6f")
    for mo in model.morphs:
        m_type = int(mo.morph_type)
        emit(_pack_text(mo.name, encoding))
        emit(_pack_text(mo.name_e, encoding))
        emit(struct.pack("<BBi", int(mo.category), m_type, len(mo.offsets)))
        if m_type in (0, 9):
            for off in mo.offsets:
                emit(gm_struct.pack(int(off.morph), float(off.factor)))
        elif m_type == 1:
            for off in mo.offsets:
                emit(
                    vm_struct.pack(
                        int(off.index),
                        float(off.offset[0]),
                        float(off.offset[1]),
                        float(off.offset[2]),
                    )
                )
        elif m_type == 2:
            for off in mo.offsets:
                emit(
                    bm_struct.pack(
                        int(off.index),
                        *(float(x) for x in off.location_offset),
                        *(float(x) for x in off.rotation_offset),
                    )
                )
        elif 3 <= m_type <= 7:
            for off in mo.offsets:
                emit(uvm_struct.pack(int(off.index), *(float(x) for x in off.offset)))
        elif m_type == 8:
            for off in mo.offsets:
                emit(
                    mm_struct.pack(
                        int(off.index),
                        int(off.offset_type),
                        *(float(x) for x in off.diffuse),
                        *(float(x) for x in off.specular),
                        float(off.shininess),
                        *(float(x) for x in off.ambient),
                        *(float(x) for x in off.edge_color),
                        float(off.edge_size),
                        *(float(x) for x in off.texture_factor),
                        *(float(x) for x in off.sphere_texture_factor),
                        *(float(x) for x in off.toon_texture_factor),
                    )
                )
        elif m_type == 10:
            for off in mo.offsets:
                emit(
                    imp_struct.pack(
                        int(off.rigid_index),
                        int(off.is_local),
                        *(float(x) for x in off.velocity),
                        *(float(x) for x in off.torque),
                    )
                )
        else:
            raise ValueError(f"Unsupported PMX morph type on save: {m_type}")

    # Display frames
    emit(struct.pack("<i", len(model.display)))
    disp_b_struct = struct.Struct("<B" + b_fmt[1])
    disp_m_struct = struct.Struct("<B" + mo_fmt[1])
    for d in model.display:
        emit(_pack_text(d.name, encoding))
        emit(_pack_text(d.name_e, encoding))
        emit(struct.pack("<Bi", 1 if d.isSpecial else 0, len(d.data)))
        for item_type, item_idx in d.data:
            if item_type == 1:
                emit(disp_m_struct.pack(1, int(item_idx)))
            else:
                emit(disp_b_struct.pack(0, int(item_idx)))

    # Rigids
    emit(struct.pack("<i", len(model.rigids)))
    rigid_struct = struct.Struct("<" + b_fmt[1] + "BH B 14f B".replace(" ", ""))
    for rb in model.rigids:
        emit(_pack_text(rb.name, encoding))
        emit(_pack_text(rb.name_e, encoding))
        emit(
            rigid_struct.pack(
                -1 if rb.bone is None else int(rb.bone),
                int(rb.collision_group_number),
                int(rb.collision_group_mask),
                int(rb.type),
                *(float(x) for x in rb.size),
                *(float(x) for x in rb.location),
                *(float(x) for x in rb.rotation),
                float(rb.mass),
                float(rb.velocity_attenuation),
                float(rb.rotation_attenuation),
                float(rb.bounce),
                float(rb.friction),
                int(rb.mode),
            )
        )

    # Joints
    emit(struct.pack("<i", len(model.joints)))
    joint_struct = struct.Struct("<B" + rb_fmt[1] * 2 + "24f")
    for j in model.joints:
        emit(_pack_text(j.name, encoding))
        emit(_pack_text(j.name_e, encoding))
        emit(
            joint_struct.pack(
                int(j.mode),
                int(j.src_rigid),
                int(j.dest_rigid),
                *(float(x) for x in j.location),
                *(float(x) for x in j.rotation),
                *(float(x) for x in j.minimum_location),
                *(float(x) for x in j.maximum_location),
                *(float(x) for x in j.minimum_rotation),
                *(float(x) for x in j.maximum_rotation),
                *(float(x) for x in j.spring_constant),
                *(float(x) for x in j.spring_rotation_constant),
            )
        )

    path.write_bytes(b"".join(chunks))
