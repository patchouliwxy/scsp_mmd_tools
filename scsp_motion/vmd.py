"""Strict VMD 0002 writer and independent structural reader."""
from pathlib import Path
import math
import struct
import numpy as np


def name_bytes(name, width):
    if not name or '\0' in name:
        raise ValueError('Empty/NUL VMD name')
    value = name.encode('cp932')
    if len(value) > width:
        raise ValueError(f'VMD name exceeds {width} CP932 bytes: {name}')
    return value.ljust(width, b'\0')


def linear_bone():
    base = bytes([20] * 8 + [107] * 8)
    return b''.join(base[i:] + bytes(i) for i in range(4))


def stable_indices(values):
    """Remove only exact constant interiors; preserve both edges of each hold."""
    values = np.asarray(values)
    if len(values) < 3:
        return np.arange(len(values))
    same = np.all(values[1:] == values[:-1], axis=tuple(range(1, values.ndim)))
    return np.flatnonzero(np.r_[True, ~(same[:-1] & same[1:]), True])


def write(path, model_name, bones=None, morphs=None, cameras=None, ik=None):
    """Tracks are (integer frames, values). Bone values: XYZ + XYZW."""
    path = Path(path)
    bones, morphs, cameras, ik = bones or {}, morphs or {}, cameras or [], ik or []
    encoded_model = name_bytes(model_name, 20)
    checked_bones = []; checked_morphs = []
    for name, (frames, values) in bones.items():
        field = name_bytes(name, 15)
        frames = validate_frames(frames); values = np.asarray(values)
        if values.shape != (len(frames), 7) or not np.isfinite(values).all():
            raise ValueError('Invalid bone values: ' + name)
        if np.max(np.abs(np.linalg.norm(values[:, 3:], axis=1) - 1)) > 1e-4:
            raise ValueError('Non-unit bone quaternion: ' + name)
        checked_bones.append((field, frames, values))
    for name, (frames, values) in morphs.items():
        field = name_bytes(name, 15); frames = validate_frames(frames); values = np.asarray(values)
        if values.shape != (len(frames),) or not np.isfinite(values).all():
            raise ValueError('Invalid morph values: ' + name)
        checked_morphs.append((field, frames, values))
    for cam in cameras:
        if not np.isfinite([cam['distance'], *cam['position'], *cam['rotation']]).all():
            raise ValueError('Invalid camera values')
        if not 1 <= cam['fov'] <= 179:
            raise ValueError('Camera FOV outside VMD range')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: output must never silently replace a user's motion.
    with path.open('xb') as out:
        out.write(b'Vocaloid Motion Data 0002'.ljust(30, b'\0')); out.write(encoded_model)
        out.write(struct.pack('<I', sum(len(x[1]) for x in checked_bones)))
        for name, frames, values in checked_bones:
            for frame, v in zip(frames, values):
                out.write(struct.pack('<15sI7f64s', name, int(frame), *v, linear_bone()))
        out.write(struct.pack('<I', sum(len(x[1]) for x in checked_morphs)))
        for name, frames, values in checked_morphs:
            for frame, weight in zip(frames, values):
                out.write(struct.pack('<15sIf', name, int(frame), float(weight)))
        out.write(struct.pack('<I', len(cameras)))
        for c in cameras:
            out.write(struct.pack('<If3f3f24sIB', int(c['frame']), c['distance'], *c['position'],
                                  *c['rotation'], c.get('interpolation', bytes([20, 107, 20, 107] * 6)),
                                  int(c['fov']), 0 if c.get('perspective', True) else 1))
        out.write(struct.pack('<II', 0, 0))  # light, self shadow
        out.write(struct.pack('<I', len(ik)))
        for frame, entries in ik:
            out.write(struct.pack('<IBI', int(frame), 1, len(entries)))
            for name, enabled in entries.items():
                out.write(struct.pack('<20sB', name_bytes(name, 20), int(enabled)))
    return inspect(path)


def validate_frames(frames):
    a = np.asarray(frames)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 0xffffffff) or np.any(a != np.floor(a)) or np.any(np.diff(a.astype(float)) <= 0):
        raise ValueError('VMD frames must be unique increasing uint32 integers')
    return a.astype(np.uint32)


def inspect(path):
    counts = {}; names = {}; last_frame = 0
    with Path(path).open('rb') as src:
        def read(n):
            b = src.read(n)
            if len(b) != n:
                raise ValueError('Truncated VMD')
            return b
        def count():
            n = struct.unpack('<I', read(4))[0]
            if n > 10000000:
                raise ValueError('Implausible VMD record count')
            return n
        header = read(30)
        if header.rstrip(b'\0') != b'Vocaloid Motion Data 0002':
            raise ValueError('Unsupported VMD header')
        model = read(20).split(b'\0')[0].decode('cp932')
        for kind, size in [('bones',111), ('morphs',23), ('cameras',61), ('lights',28), ('shadows',9)]:
            n = count(); counts[kind] = n; found = set(); named = kind in ('bones', 'morphs')
            for _ in range(n):
                data = read(size); named = kind in ('bones', 'morphs'); off = 15 if named else 0
                frame = struct.unpack_from('<I', data, off)[0]; last_frame = max(last_frame, frame)
                if named:
                    found.add(data[:15].split(b'\0')[0].decode('cp932'))
                if kind == 'bones':
                    numbers = struct.unpack_from('<7f', data, 19)
                    if not all(math.isfinite(x) for x in numbers) or abs(sum(x*x for x in numbers[3:])-1) > 1e-3:
                        raise ValueError('Invalid saved bone record')
            if named:
                names[kind] = sorted(found)
        counts['ik'] = count()
        for _ in range(counts['ik']):
            frame, visible, n = struct.unpack('<IBI', read(9)); last_frame = max(last_frame, frame)
            for _ in range(n):
                read(20).split(b'\0')[0].decode('cp932'); read(1)
        if src.read(1):
            raise ValueError('Unexpected VMD trailing bytes')
    return dict(model=model, counts=counts, names=names, last_frame=last_frame)
