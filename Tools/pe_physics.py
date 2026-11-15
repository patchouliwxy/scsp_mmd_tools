"""Independent MMD native physics post-processor (pmx.Rigid + pmx.Joint).

Reads a clean, physics-free PMX produced by the main studio pipeline and
writes a separate `<Name>_physics.pmx` without modifying any existing
pipeline scripts, `mmd_tools` extension files, vertex weights, or bone layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from mathutils import Euler, Matrix, Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmx_io as pmx


# Standard MMD / PE bones that are never treated as dangling physics chains.
STANDARD_BONES = {
    "全ての親", "操作中心", "センター", "グルーブ", "腰",
    "下半身", "上半身", "上半身2", "上半身3", "首", "頭", "頭先",
    "両目", "左目", "右目",
    "左肩", "左肩P", "左肩C", "左腕", "左腕捩", "左腕捩1", "左腕捩2", "左腕捩3",
    "左ひじ", "左手捩", "左手捩1", "左手捩2", "左手捩3", "左手首",
    "右肩", "右肩P", "右肩C", "右腕", "右腕捩", "右腕捩1", "右腕捩2", "右腕捩3",
    "右ひじ", "右手捩", "右手捩1", "右手捩2", "右手捩3", "右手首",
    "左足", "左ひざ", "左足首", "左つま先", "左つま先先", "左足先", "左足ＩＫ", "左足IK親", "左つま先ＩＫ",
    "右足", "右ひざ", "右足首", "右つま先", "右つま先先", "右足先", "右足ＩＫ", "右足IK親", "右つま先ＩＫ",
}

FINGER_TOKENS = ("親指", "人指", "中指", "薬指", "小指")
STRUCTURAL_EXTRA_NAMES = {
    "LeftClavicle", "RightClavicle",
    "LeftRing", "RightRing",
    "LeftToe_null", "RightToe_null",
}
CONTAINER_BONE_NAMES = {
    "p02__hair_boneGp", "p02__HAIR",
    "p03__acce_boneGp", "p03__acce",
    "cloth_group",
}

# 0-based collision group indices (PMX 1..16 -> 0..15)
GRP_BODY_UPPER = 0
GRP_BODY_LOWER = 1
GRP_HAIR_FRONT = 2
GRP_HAIR_BACK = 3
GRP_SKIRT = 4
GRP_CLOTH = 5
GRP_BUST = 6


def non_collision_mask(collide_with: set[int]) -> int:
    """Return 16-bit PMX non-collision mask (bit i == 0 collides, 1 ignores)."""
    mask = 0xFFFF
    for g in collide_with:
        if 0 <= g < 16:
            mask &= ~(1 << g)
    return mask


def _is_finite_seq(seq: Any) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in seq)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _pmx_rot_from_dir(dir_pmx: Vector) -> list[float]:
    """Compute pmx.Rigid/Joint YXZ rotation aligning capsule local Y to dir_pmx."""
    if dir_pmx.length < 1e-6:
        return [0.0, 0.0, 0.0]
    # In mmd_tools, PMX (x, y, z) -> Blender (x, z, y), and capsule local axis is Blender +Z.
    d_bl = Vector((dir_pmx.x, dir_pmx.z, dir_pmx.y)).normalized()
    q_bl = Vector((0.0, 0.0, 1.0)).rotation_difference(d_bl)
    e_bl = q_bl.to_euler("YXZ")
    # Exporter maps Blender Euler -> PMX via Vector(e_bl).xzy * -1.0
    return [float(-e_bl[0]), float(-e_bl[2]), float(-e_bl[1])]


def _collect_bone_vertices(model: pmx.Model) -> dict[int, list[Vector]]:
    bone_verts: dict[int, list[Vector]] = {}
    for v in model.vertices:
        w = v.weight
        co = Vector(v.co)
        if hasattr(w, "weights"):
            for bi, wt in zip(w.bones, w.weights):
                if bi is not None and bi >= 0 and wt > 0.05:
                    bone_verts.setdefault(bi, []).append(co)
        elif w.bones:
            bi = w.bones[0]
            if bi is not None and bi >= 0:
                bone_verts.setdefault(bi, []).append(co)
    return bone_verts


def _fit_segment_radius(
    p0: Vector,
    p1: Vector,
    verts: list[Vector],
    default_radius: float,
    min_r: float,
    max_r: float,
    percentile: float = 0.80,
) -> float:
    if len(verts) < 8:
        return _clamp(default_radius, min_r, max_r)
    seg = p1 - p0
    seg_len = seg.length
    if seg_len < 1e-5:
        dists = sorted((v - p0).length for v in verts)
    else:
        axis = seg / seg_len
        dists = []
        for v in verts:
            rel = v - p0
            proj = rel.dot(axis)
            if -0.25 * seg_len <= proj <= 1.25 * seg_len:
                perp = (rel - axis * proj).length
                dists.append(perp)
        if len(dists) < 4:
            dists = [(v - p0).length for v in verts]
        dists.sort()
    idx = min(len(dists) - 1, max(0, int(len(dists) * percentile)))
    return _clamp(dists[idx], min_r, max_r)


def _make_rigid(
    name: str,
    bone_idx: int,
    mode: int,
    shape_type: int,
    size: list[float],
    location: list[float],
    rotation: list[float],
    group_num: int,
    collide_with: set[int],
    mass: float = 1.0,
    vel_att: float = 0.5,
    rot_att: float = 0.5,
    bounce: float = 0.0,
    friction: float = 0.5,
) -> pmx.Rigid:
    r = pmx.Rigid()
    r.name = name
    r.name_e = name
    r.bone = bone_idx
    r.mode = mode
    r.type = shape_type
    r.size = [float(x) for x in size]
    r.location = [float(x) for x in location]
    r.rotation = [float(x) for x in rotation]
    r.collision_group_number = group_num
    r.collision_group_mask = non_collision_mask(collide_with)
    r.mass = float(mass)
    r.velocity_attenuation = float(vel_att)
    r.rotation_attenuation = float(rot_att)
    r.bounce = float(bounce)
    r.friction = float(friction)
    return r


def _make_joint(
    name: str,
    src_rigid: int,
    dest_rigid: int,
    location: list[float],
    rotation: list[float],
    min_loc: list[float],
    max_loc: list[float],
    min_rot: list[float],
    max_rot: list[float],
    spring_loc: list[float],
    spring_rot: list[float],
) -> pmx.Joint:
    j = pmx.Joint()
    j.name = name
    j.name_e = name
    j.mode = pmx.Joint.MODE_SPRING6DOF
    j.src_rigid = src_rigid
    j.dest_rigid = dest_rigid
    j.location = [float(x) for x in location]
    j.rotation = [float(x) for x in rotation]
    j.minimum_location = [float(x) for x in min_loc]
    j.maximum_location = [float(x) for x in max_loc]
    j.minimum_rotation = [float(x) for x in min_rot]
    j.maximum_rotation = [float(x) for x in max_rot]
    j.spring_constant = [float(x) for x in spring_loc]
    j.spring_rotation_constant = [float(x) for x in spring_rot]
    return j


def _is_standard_or_structural(name: str) -> bool:
    if name in STANDARD_BONES or name in STRUCTURAL_EXTRA_NAMES:
        return True
    if any(tok in name for tok in FINGER_TOKENS):
        return True
    return False


def _classify_chain(root_name: str) -> str:
    low = root_name.lower()
    if "hair" in low:
        if any(k in low for k in ("front", "side", "thin")):
            return "hair_front"
        return "hair_back"
    if any(k in low for k in ("skirt", "outer", "jacket", "coat")):
        return "skirt"
    return "cloth"


def _build_body_colliders(
    model: pmx.Model,
    bone_map: dict[str, int],
    bone_verts: dict[int, list[Vector]],
    scale: float,
) -> tuple[list[pmx.Rigid], dict[int, int]]:
    rigids: list[pmx.Rigid] = []
    anchor_by_bone: dict[int, int] = {}

    def add_seg_capsule(
        r_name: str,
        b_start: str,
        b_end: str,
        group: int,
        collide_with: set[int],
        default_r: float,
        min_r: float,
        max_r: float,
        shrink: float = 0.92,
        record_anchor: bool = True,
    ) -> None:
        if b_start not in bone_map or b_end not in bone_map:
            return
        i0, i1 = bone_map[b_start], bone_map[b_end]
        p0 = Vector(model.bones[i0].location)
        p1 = Vector(model.bones[i1].location)
        seg = p1 - p0
        if seg.length < 1e-4:
            return
        verts = bone_verts.get(i0, [])
        rad = _fit_segment_radius(p0, p1, verts, default_r * scale, min_r * scale, max_r * scale) * shrink
        cyl_h = max(0.05 * scale, seg.length - 0.35 * rad)
        mid = (p0 + p1) * 0.5
        rot = _pmx_rot_from_dir(seg)
        idx = len(rigids)
        rigids.append(
            _make_rigid(
                name=r_name,
                bone_idx=i0,
                mode=pmx.Rigid.MODE_STATIC,
                shape_type=pmx.Rigid.TYPE_CAPSULE,
                size=[rad, cyl_h, 0.0],
                location=list(mid),
                rotation=rot,
                group_num=group,
                collide_with=collide_with,
                mass=1.0,
                vel_att=0.0,
                rot_att=0.0,
                bounce=0.0,
                friction=0.5,
            )
        )
        if record_anchor and i0 not in anchor_by_bone:
            anchor_by_bone[i0] = idx

    # Head sphere collider (fitted slightly behind/above head pivot to protect skull/cheeks)
    if "頭" in bone_map:
        hi = bone_map["頭"]
        hp = Vector(model.bones[hi].location)
        hverts = bone_verts.get(hi, [])
        if "頭先" in bone_map:
            htip = Vector(model.bones[bone_map["頭先"]].location)
        else:
            htip = hp + Vector((0.0, 2.2 * scale, 0.0))
        rad = _fit_segment_radius(hp, htip, hverts, 1.05 * scale, 0.85 * scale, 1.45 * scale, percentile=0.70) * 0.72
        center = (hp + htip) * 0.5 + Vector((0.0, 0.0, 0.25 * scale))
        idx = len(rigids)
        rigids.append(
            _make_rigid(
                name="Col_Head",
                bone_idx=hi,
                mode=pmx.Rigid.MODE_STATIC,
                shape_type=pmx.Rigid.TYPE_SPHERE,
                size=[rad, 0.0, 0.0],
                location=list(center),
                rotation=[0.0, 0.0, 0.0],
                group_num=GRP_BODY_UPPER,
                collide_with={GRP_HAIR_FRONT, GRP_HAIR_BACK, GRP_CLOTH},
            )
        )
        anchor_by_bone[hi] = idx

    add_seg_capsule("Col_Neck", "首", "頭", GRP_BODY_UPPER, {GRP_HAIR_BACK, GRP_CLOTH}, 0.50, 0.35, 0.70, 0.88)
    add_seg_capsule("Col_Upper1", "上半身", "上半身2", GRP_BODY_UPPER, {GRP_HAIR_BACK, GRP_CLOTH}, 1.05, 0.75, 1.45, 0.86)
    add_seg_capsule("Col_Upper2", "上半身2", "上半身3", GRP_BODY_UPPER, {GRP_HAIR_BACK, GRP_CLOTH}, 1.15, 0.80, 1.55, 0.86)
    add_seg_capsule("Col_Upper3", "上半身3", "首", GRP_BODY_UPPER, {GRP_HAIR_BACK, GRP_CLOTH}, 1.10, 0.75, 1.50, 0.84)

    # Arms
    for side, pfx in (("L", "左"), ("R", "右")):
        add_seg_capsule(f"Col_Arm_{side}", f"{pfx}腕", f"{pfx}ひじ", GRP_BODY_UPPER, {GRP_HAIR_BACK, GRP_CLOTH}, 0.42, 0.28, 0.65, 0.88)
        add_seg_capsule(f"Col_ForeArm_{side}", f"{pfx}ひじ", f"{pfx}手首", GRP_BODY_UPPER, {GRP_HAIR_BACK, GRP_CLOTH}, 0.35, 0.22, 0.55, 0.88)

    # Lower body / pelvis + legs (Group 1 -> Skirt & Coat collisions)
    if "下半身" in bone_map:
        li = bone_map["下半身"]
        lp = Vector(model.bones[li].location)
        lverts = bone_verts.get(li, [])
        rad = _fit_segment_radius(lp, lp + Vector((0.0, 1.0 * scale, 0.0)), lverts, 1.05 * scale, 0.75 * scale, 1.45 * scale, percentile=0.65) * 0.78
        idx = len(rigids)
        rigids.append(
            _make_rigid(
                name="Col_Lower",
                bone_idx=li,
                mode=pmx.Rigid.MODE_STATIC,
                shape_type=pmx.Rigid.TYPE_SPHERE,
                size=[rad, 0.0, 0.0],
                location=list(lp),
                rotation=[0.0, 0.0, 0.0],
                group_num=GRP_BODY_LOWER,
                collide_with={GRP_SKIRT},
            )
        )
        anchor_by_bone[li] = idx

    for side, pfx in (("L", "左"), ("R", "右")):
        add_seg_capsule(f"Col_Thigh_{side}", f"{pfx}足", f"{pfx}ひざ", GRP_BODY_LOWER, {GRP_SKIRT}, 0.65, 0.45, 0.95, 0.82)
        add_seg_capsule(f"Col_Shin_{side}", f"{pfx}ひざ", f"{pfx}足首", GRP_BODY_LOWER, {GRP_SKIRT}, 0.48, 0.32, 0.72, 0.85)

    return rigids, anchor_by_bone


def _point_to_segment_dist(p: Vector, a: Vector, b: Vector) -> float:
    ab = b - a
    len_sq = ab.length_squared
    if len_sq < 1e-10:
        return (p - a).length
    t = _clamp((p - a).dot(ab) / len_sq, 0.0, 1.0)
    proj = a + ab * t
    return (p - proj).length


def _capsule_endpoints_pmx(r: pmx.Rigid) -> tuple[Vector, Vector, float]:
    c = Vector(r.location)
    rad = float(r.size[0])
    if r.type != pmx.Rigid.TYPE_CAPSULE:
        return c, c, rad
    half_h = float(r.size[1]) * 0.5
    e_bl = Euler((-r.rotation[0], -r.rotation[2], -r.rotation[1]), "YXZ")
    z_bl = e_bl.to_matrix() @ Vector((0.0, 0.0, 1.0))
    axis_pmx = Vector((z_bl.x, z_bl.z, z_bl.y)).normalized()
    return c - axis_pmx * half_h, c + axis_pmx * half_h, rad


def _segment_to_segment_dist(a0: Vector, a1: Vector, b0: Vector, b1: Vector) -> float:
    # Approximate exact segment-segment distance by sampling endpoints and midpoints
    pts_a = (a0, (a0 + a1) * 0.5, a1)
    pts_b = (b0, (b0 + b1) * 0.5, b1)
    d = min(_point_to_segment_dist(pa, b0, b1) for pa in pts_a)
    d = min(d, min(_point_to_segment_dist(pb, a0, a1) for pb in pts_b))
    return d


def _enforce_frame0_clearance(rigids: list[pmx.Rigid], scale: float) -> None:
    """Shrink static colliders if needed so frame-0 rest pose has no initial penetration."""
    gap = 0.08 * scale
    min_rad = 0.15 * scale
    dyn_rigids = [r for r in rigids if r.mode != pmx.Rigid.MODE_STATIC and r.collision_group_mask != 0xFFFF]
    dyn_segs = [(r, *_capsule_endpoints_pmx(r)) for r in dyn_rigids]

    for sr in rigids:
        if sr.mode != pmx.Rigid.MODE_STATIC or sr.collision_group_mask == 0xFFFF:
            continue
        sa0, sa1, s_rad = _capsule_endpoints_pmx(sr)
        max_allowed = s_rad
        for dr, da0, da1, d_rad in dyn_segs:
            # Check mutual collision: bit dr.group in sr.mask == 0 AND bit sr.group in dr.mask == 0
            sr_hits_dr = (sr.collision_group_mask & (1 << dr.collision_group_number)) == 0
            dr_hits_sr = (dr.collision_group_mask & (1 << sr.collision_group_number)) == 0
            if not (sr_hits_dr and dr_hits_sr):
                continue
            dist = _segment_to_segment_dist(sa0, sa1, da0, da1)
            allowed = dist - d_rad - gap
            if allowed < max_allowed:
                max_allowed = allowed
        if max_allowed < s_rad:
            if max_allowed >= min_rad:
                sr.size[0] = float(max_allowed)
            else:
                # If dynamic chain starts directly inside this static collider, disable collision on this collider
                sr.size[0] = float(min_rad)


def _ensure_static_anchor(
    model: pmx.Model,
    bone_idx: int,
    rigids: list[pmx.Rigid],
    anchor_by_bone: dict[int, int],
    scale: float,
) -> int:
    if bone_idx in anchor_by_bone:
        return anchor_by_bone[bone_idx]
    b = model.bones[bone_idx]
    idx = len(rigids)
    rigids.append(
        _make_rigid(
            name=f"Anchor_{b.name}",
            bone_idx=bone_idx,
            mode=pmx.Rigid.MODE_STATIC,
            shape_type=pmx.Rigid.TYPE_SPHERE,
            size=[0.20 * scale, 0.0, 0.0],
            location=list(b.location),
            rotation=[0.0, 0.0, 0.0],
            group_num=GRP_BODY_UPPER,
            collide_with=set(),
        )
    )
    anchor_by_bone[bone_idx] = idx
    return idx


def _effective_anchor_bone(model: pmx.Model, bone_idx: int) -> int:
    """Walk up past zero-weight container bones to find a solid parent bone."""
    cur = model.bones[bone_idx].parent
    while cur is not None and 0 <= cur < len(model.bones):
        cname = model.bones[cur].name
        if cname in CONTAINER_BONE_NAMES:
            cur = model.bones[cur].parent
            continue
        return cur
    return 0


def _discover_dynamic_chains(
    model: pmx.Model,
    children_map: dict[int, list[int]],
    bone_verts: dict[int, list[Vector]],
) -> tuple[list[tuple[int, list[int]]], list[tuple[int, int]]]:
    """Return (linear_chains, bust_pairs) where each chain is (anchor_bone_idx, [bone_indices])."""
    chains: list[tuple[int, list[int]]] = []
    bust_pairs: list[tuple[int, int]] = []
    visited: set[int] = set()

    def walk_chain(start_idx: int, anchor_bone: int) -> None:
        if start_idx in visited:
            return
        cur = start_idx
        seq: list[int] = []
        while cur is not None and cur not in visited:
            visited.add(cur)
            bname = model.bones[cur].name
            if _is_standard_or_structural(bname):
                break
            if bname in CONTAINER_BONE_NAMES:
                for ch in children_map.get(cur, []):
                    walk_chain(ch, anchor_bone)
                return
            seq.append(cur)
            kids = [c for c in children_map.get(cur, []) if not _is_standard_or_structural(model.bones[c].name)]
            if len(kids) == 1:
                cur = kids[0]
            else:
                # Branching: emit current segment, then start child chains anchored to cur
                for ch in kids:
                    walk_chain(ch, cur)
                break
        if seq:
            # Skip single leaf rigid mount markers (*_pos) or completely empty zero-length chains
            if len(seq) == 1 and model.bones[seq[0]].name.endswith("_pos"):
                return
            has_weights = any(len(bone_verts.get(bi, [])) > 0 for bi in seq)
            if has_weights or len(seq) >= 2:
                chains.append((anchor_bone, seq))

    for i, b in enumerate(model.bones):
        if _is_standard_or_structural(b.name):
            continue
        if b.name == "p00__face_boneGp":
            # Mark entire face subtree visited so facial bones are never touched
            stack = [i]
            while stack:
                n = stack.pop()
                visited.add(n)
                stack.extend(children_map.get(n, []))
            continue

    # Detect OPAI_L0->OPAI_L1 and OPAI_R0->OPAI_R1
    for side in ("L", "R"):
        r0_name = f"OPAI_{side}0"
        r1_name = f"OPAI_{side}1"
        r0 = next((i for i, b in enumerate(model.bones) if b.name == r0_name), None)
        r1 = next((i for i, b in enumerate(model.bones) if b.name == r1_name), None)
        if r0 is not None and r1 is not None:
            visited.add(r0)
            visited.add(r1)
            bust_pairs.append((r0, r1))

    # Walk from all standard bones into non-standard subtrees
    for i, b in enumerate(model.bones):
        if not _is_standard_or_structural(b.name):
            continue
        for ch in children_map.get(i, []):
            ch_name = model.bones[ch].name
            if ch in visited or _is_standard_or_structural(ch_name):
                continue
            if ch_name in CONTAINER_BONE_NAMES:
                anchor = _effective_anchor_bone(model, ch)
                for sub in children_map.get(ch, []):
                    walk_chain(sub, anchor)
            else:
                walk_chain(ch, i)

    return chains, bust_pairs


def build_physics(model: pmx.Model) -> tuple[pmx.Model, dict[str, Any]]:
    """Populate pmx.Rigid and pmx.Joint on a PMX model without altering geometry/weights."""
    bone_map = {b.name: i for i, b in enumerate(model.bones)}
    children_map: dict[int, list[int]] = {}
    for i, b in enumerate(model.bones):
        if b.parent is not None and 0 <= b.parent < len(model.bones):
            children_map.setdefault(b.parent, []).append(i)

    ys = [v.co[1] for v in model.vertices]
    model_height = (max(ys) - min(ys)) if ys else 20.0
    scale = max(0.25, model_height / 20.0)
    bone_verts = _collect_bone_vertices(model)

    rigids, anchor_by_bone = _build_body_colliders(model, bone_map, bone_verts, scale)
    joints: list[pmx.Joint] = []

    chains, bust_pairs = _discover_dynamic_chains(model, children_map, bone_verts)

    # 1. Bust (OPAI_L1 / OPAI_R1)
    for r0, r1 in bust_pairs:
        anchor_rigid = _ensure_static_anchor(model, r0, rigids, anchor_by_bone, scale)
        p1 = Vector(model.bones[r1].location)
        verts = bone_verts.get(r1, [])
        rad = _fit_segment_radius(p1, p1, verts, 0.55 * scale, 0.35 * scale, 0.80 * scale, percentile=0.60) * 0.70
        dest_idx = len(rigids)
        rigids.append(
            _make_rigid(
                name=f"Phy_{model.bones[r1].name}",
                bone_idx=r1,
                mode=pmx.Rigid.MODE_DYNAMIC_BONE,
                shape_type=pmx.Rigid.TYPE_SPHERE,
                size=[rad, 0.0, 0.0],
                location=list(p1),
                rotation=[0.0, 0.0, 0.0],
                group_num=GRP_BUST,
                collide_with=set(),
                mass=0.6,
                vel_att=0.995,
                rot_att=0.999,
                bounce=0.0,
                friction=0.2,
            )
        )
        joints.append(
            _make_joint(
                name=f"J_{model.bones[r1].name}",
                src_rigid=anchor_rigid,
                dest_rigid=dest_idx,
                location=list(p1),
                rotation=[0.0, 0.0, 0.0],
                min_loc=[-0.04 * scale, -0.05 * scale, -0.03 * scale],
                max_loc=[0.04 * scale, 0.05 * scale, 0.04 * scale],
                min_rot=[-0.18, -0.10, -0.15],
                max_rot=[0.18, 0.10, 0.15],
                spring_loc=[8.0, 10.0, 8.0],
                spring_rot=[4.0, 4.0, 4.0],
            )
        )

    # 2. Dynamic chains (hair, skirt, cloth)
    chain_rigids_by_cat: dict[str, list[list[int]]] = {"hair_front": [], "hair_back": [], "skirt": [], "cloth": []}
    pelvis_center = Vector(model.bones[bone_map["下半身"]].location) if "下半身" in bone_map else Vector((0.0, 10.0 * scale, 0.0))

    for anchor_bone, seq in chains:
        cat = _classify_chain(model.bones[seq[0]].name)
        parent_rigid = _ensure_static_anchor(model, anchor_bone, rigids, anchor_by_bone, scale)
        seg_rigids: list[int] = []

        if cat == "hair_front":
            grp, collide = GRP_HAIR_FRONT, {GRP_BODY_UPPER}
            swing, twist, vel_att, rot_att, spr_rot = 0.18, 0.06, 0.995, 0.999, 3.0
        elif cat == "hair_back":
            grp, collide = GRP_HAIR_BACK, {GRP_BODY_UPPER}
            swing, twist, vel_att, rot_att, spr_rot = 0.45, 0.12, 0.995, 0.999, 1.5
        elif cat == "skirt":
            grp, collide = GRP_SKIRT, {GRP_BODY_LOWER}
            swing, twist, vel_att, rot_att, spr_rot = 0.48, 0.10, 0.995, 0.999, 1.5
        else:
            grp, collide = GRP_CLOTH, {GRP_BODY_UPPER}
            swing, twist, vel_att, rot_att, spr_rot = 0.42, 0.10, 0.995, 0.999, 2.0

        prev_dir = Vector((0.0, -1.0, 0.0))
        prev_len = 0.5 * scale

        for idx_in_chain, bi in enumerate(seq):
            b = model.bones[bi]
            p0 = Vector(b.location)
            if idx_in_chain + 1 < len(seq):
                p1 = Vector(model.bones[seq[idx_in_chain + 1]].location)
            elif isinstance(b.displayConnection, (list, tuple)) and len(b.displayConnection) == 3 and Vector(b.displayConnection).length > 1e-4:
                p1 = p0 + Vector(b.displayConnection)
            elif idx_in_chain > 0:
                p1 = p0 + prev_dir * prev_len
            else:
                p1 = p0 + Vector((0.0, -0.5 * scale, 0.0))

            seg = p1 - p0
            seg_len = max(0.06 * scale, seg.length)
            seg_dir = seg.normalized() if seg.length > 1e-5 else prev_dir
            prev_dir = seg_dir
            prev_len = seg_len

            rad = _clamp(0.20 * seg_len, 0.08 * scale, 0.26 * scale)
            cyl_h = max(0.04 * scale, seg_len - 0.5 * rad)
            mid = (p0 + p1) * 0.5
            rot = _pmx_rot_from_dir(seg_dir)
            mass = max(0.25, 1.0 - 0.15 * idx_in_chain)

            r_idx = len(rigids)
            rigids.append(
                _make_rigid(
                    name=f"Phy_{b.name}",
                    bone_idx=bi,
                    mode=pmx.Rigid.MODE_DYNAMIC,
                    shape_type=pmx.Rigid.TYPE_CAPSULE,
                    size=[rad, cyl_h, 0.0],
                    location=list(mid),
                    rotation=rot,
                    group_num=grp,
                    collide_with=collide,
                    mass=mass,
                    vel_att=vel_att,
                    rot_att=rot_att,
                    bounce=0.0,
                    friction=0.2,
                )
            )
            anchor_by_bone.setdefault(bi, r_idx)
            seg_rigids.append(r_idx)

            tier_swing = swing * (0.70 if (cat == "skirt" and idx_in_chain == 0) else (1.0 + 0.12 * idx_in_chain))
            tier_swing = min(0.85, tier_swing)
            joints.append(
                _make_joint(
                    name=f"J_{b.name}",
                    src_rigid=parent_rigid,
                    dest_rigid=r_idx,
                    location=list(p0),
                    rotation=rot,
                    min_loc=[0.0, 0.0, 0.0],
                    max_loc=[0.0, 0.0, 0.0],
                    min_rot=[-tier_swing, -twist, -tier_swing],
                    max_rot=[tier_swing, twist, tier_swing],
                    spring_loc=[0.0, 0.0, 0.0],
                    spring_rot=[spr_rot, spr_rot * 0.5, spr_rot],
                )
            )
            parent_rigid = r_idx

        chain_rigids_by_cat[cat].append(seg_rigids)

    # 3. Lateral ring joints for Skirt / Coat / Outer chains
    skirt_chains = chain_rigids_by_cat["skirt"]
    if len(skirt_chains) >= 3:
        max_len = max(len(c) for c in skirt_chains)
        for tier in range(max_len):
            ring: list[tuple[float, int]] = []
            for c in skirt_chains:
                if tier < len(c):
                    r_idx = c[tier]
                    pos = Vector(rigids[r_idx].location)
                    ang = math.atan2(pos.z - pelvis_center.z, pos.x - pelvis_center.x)
                    ring.append((ang, r_idx))
            if len(ring) < 3:
                continue
            ring.sort(key=lambda item: item[0])
            for k in range(len(ring)):
                ang_a, ra = ring[k]
                ang_b, rb = ring[(k + 1) % len(ring)]
                d_ang = (ang_b - ang_a) % (2.0 * math.pi)
                # Only connect adjacent panels within 90 degrees so open front coats stay open
                if d_ang < (math.pi * 0.5):
                    mid = (Vector(rigids[ra].location) + Vector(rigids[rb].location)) * 0.5
                    joints.append(
                        _make_joint(
                            name=f"J_SkirtRing_T{tier}_{k}",
                            src_rigid=ra,
                            dest_rigid=rb,
                            location=list(mid),
                            rotation=[0.0, 0.0, 0.0],
                            min_loc=[-0.12 * scale, -0.12 * scale, -0.12 * scale],
                            max_loc=[0.12 * scale, 0.12 * scale, 0.12 * scale],
                            min_rot=[-0.50, -0.50, -0.50],
                            max_rot=[0.50, 0.50, 0.50],
                            spring_loc=[0.0, 0.0, 0.0],
                            spring_rot=[0.0, 0.0, 0.0],
                        )
                    )

    _enforce_frame0_clearance(rigids, scale)

    model.rigids = rigids
    model.joints = joints

    summary = {
        "model_height_pmx": round(model_height, 4),
        "scale": round(scale, 4),
        "rigid_count": len(rigids),
        "static_rigid_count": sum(1 for r in rigids if r.mode == pmx.Rigid.MODE_STATIC),
        "dynamic_rigid_count": sum(1 for r in rigids if r.mode != pmx.Rigid.MODE_STATIC),
        "joint_count": len(joints),
        "bust_count": len(bust_pairs),
        "chains_by_category": {k: len(v) for k, v in chain_rigids_by_cat.items()},
    }
    return model, summary


def audit_physics_pmx(input_model: pmx.Model, output_model: pmx.Model) -> dict[str, Any]:
    """Verify structural integrity of the generated physics PMX against input_model."""
    errors: list[str] = []

    if len(output_model.vertices) != len(input_model.vertices):
        errors.append("Vertex count changed")
    if len(output_model.faces) != len(input_model.faces):
        errors.append("Face count changed")
    if len(output_model.materials) != len(input_model.materials):
        errors.append("Material count changed")
    if len(output_model.morphs) != len(input_model.morphs):
        errors.append("Morph count changed")
    if [b.name for b in output_model.bones] != [b.name for b in input_model.bones]:
        errors.append("Bone names or order changed")
    if [b.parent for b in output_model.bones] != [b.parent for b in input_model.bones]:
        errors.append("Bone parents changed")

    num_bones = len(output_model.bones)
    num_rigids = len(output_model.rigids)
    if num_rigids == 0:
        errors.append("No rigid bodies generated")

    seen_rigid_names: set[str] = set()
    for idx, r in enumerate(output_model.rigids):
        if not r.name or r.name in seen_rigid_names:
            errors.append(f"Invalid or duplicate rigid name at {idx}: {r.name!r}")
        seen_rigid_names.add(r.name)
        if r.bone is None or not (0 <= r.bone < num_bones):
            errors.append(f"Invalid rigid.bone index at {idx}: {r.bone}")
        if not (_is_finite_seq(r.location) and _is_finite_seq(r.rotation) and _is_finite_seq(r.size)):
            errors.append(f"Non-finite transform/size in rigid {idx}: {r.name}")
        if r.mass <= 0 or not math.isfinite(r.mass):
            errors.append(f"Invalid mass in rigid {idx}: {r.mass}")
        if r.size[0] <= 0 or (r.type == pmx.Rigid.TYPE_CAPSULE and r.size[1] <= 0):
            errors.append(f"Invalid size in rigid {idx}: {r.size}")

    seen_joint_names: set[str] = set()
    for idx, j in enumerate(output_model.joints):
        if not j.name or j.name in seen_joint_names:
            errors.append(f"Invalid or duplicate joint name at {idx}: {j.name!r}")
        seen_joint_names.add(j.name)
        if not (0 <= j.src_rigid < num_rigids) or not (0 <= j.dest_rigid < num_rigids):
            errors.append(f"Invalid joint rigid indices at {idx}: src={j.src_rigid}, dest={j.dest_rigid}")
        if j.src_rigid == j.dest_rigid:
            errors.append(f"Self-referencing joint at {idx}: {j.name}")
        if not (
            _is_finite_seq(j.location)
            and _is_finite_seq(j.rotation)
            and _is_finite_seq(j.minimum_location)
            and _is_finite_seq(j.maximum_location)
            and _is_finite_seq(j.minimum_rotation)
            and _is_finite_seq(j.maximum_rotation)
            and _is_finite_seq(j.spring_constant)
            and _is_finite_seq(j.spring_rotation_constant)
        ):
            errors.append(f"Non-finite values in joint {idx}: {j.name}")

    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:50],
        "rigids": num_rigids,
        "joints": len(output_model.joints),
    }


def _copy_textures_if_needed(src_pmx: Path, dst_pmx: Path) -> None:
    if src_pmx.parent.resolve() == dst_pmx.parent.resolve():
        return
    src_tex = src_pmx.parent / "textures"
    dst_tex = dst_pmx.parent / "textures"
    if src_tex.is_dir() and not dst_tex.exists():
        shutil.copytree(src_tex, dst_tex)


def process_pmx_file(input_path: Path, output_path: Path, report_path: Path | None = None) -> dict[str, Any]:
    input_model = pmx.load(str(input_path))
    work_model = pmx.load(str(input_path))
    work_model, summary = build_physics(work_model)
    audit = audit_physics_pmx(input_model, work_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    add_uv_count = max((len(v.additional_uvs) for v in work_model.vertices), default=0)
    pmx.save(str(output_path), work_model, add_uv_count=add_uv_count)
    _copy_textures_if_needed(input_path, output_path)

    # Re-load saved PMX to confirm binary round-trip integrity
    reloaded = pmx.load(str(output_path))
    roundtrip_audit = audit_physics_pmx(input_model, reloaded)

    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "summary": summary,
        "audit": audit,
        "roundtrip_audit": roundtrip_audit,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["passed"] or not roundtrip_audit["passed"]:
        raise RuntimeError(f"Physics audit failed for {input_path.name}: {audit['errors'] or roundtrip_audit['errors']}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add MMD native physics (Rigid + Joint) to a clean PMX.")
    parser.add_argument("--input", type=Path, help="Input clean PMX file")
    parser.add_argument("--output", type=Path, help="Output physics PMX file")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--input-dir", type=Path, help="Batch mode: scan <dir>/*/*.pmx (skipping _work)")
    parser.add_argument("--output-dir", type=Path, help="Batch mode: output directory")
    args = parser.parse_args(argv)

    if args.input and args.output:
        payload = process_pmx_file(args.input.resolve(), args.output.resolve(), args.report.resolve() if args.report else None)
        print(json.dumps(payload["summary"], ensure_ascii=False))
        return 0

    if args.input_dir and args.output_dir:
        in_dir = args.input_dir.resolve()
        out_dir = args.output_dir.resolve()
        results = []
        for pmx_file in sorted(in_dir.glob("*/*.pmx")):
            if "_work" in pmx_file.parts:
                continue
            stem = pmx_file.stem
            dst_pmx = out_dir / stem / f"{stem}_physics.pmx"
            dst_rep = out_dir / stem / "physics-report.json"
            res = process_pmx_file(pmx_file, dst_pmx, dst_rep)
            results.append({"model": stem, **res["summary"], "passed": res["roundtrip_audit"]["passed"]})
        print(json.dumps({"processed": len(results), "results": results}, ensure_ascii=False, indent=2))
        return 0

    parser.error("Provide either --input & --output, or --input-dir & --output-dir")
    return 2


if __name__ == "__main__":
    if "--" in sys.argv:
        cli_args = sys.argv[sys.argv.index("--") + 1 :]
    else:
        cli_args = sys.argv[1:]
    raise SystemExit(main(cli_args))
