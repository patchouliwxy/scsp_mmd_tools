"""Vectorized right-handed quaternion algebra; storage is x,y,z,w."""
import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-12) or not np.isfinite(v).all():
        raise ValueError("Zero-length or non-finite vector/quaternion")
    return v / norm


def inv(q):
    return np.asarray(q) * np.array([-1, -1, -1, 1])


def mul(a, b):
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    av, bv = a[..., :3], b[..., :3]
    return np.concatenate((a[..., 3:] * bv + b[..., 3:] * av + np.cross(av, bv),
                           a[..., 3:] * b[..., 3:] - np.sum(av * bv, axis=-1, keepdims=True)), axis=-1)


def rotate(q, v):
    q, v = np.asarray(q), np.asarray(v)
    t = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:] * t + np.cross(q[..., :3], t)


def between(a, b):
    a, b = unit(a), unit(b)
    dot = np.clip(np.dot(a, b), -1, 1)
    if dot < -0.999999:
        axis = np.cross(a, [1., 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0., 1, 0])
        return np.r_[unit(axis), 0.]
    return unit(np.r_[np.cross(a, b), 1 + dot])


def continuous(q):
    q = unit(q).copy()
    if len(q) > 1:
        signs = np.where(np.sum(q[1:] * q[:-1], axis=-1) < 0, -1, 1)
        q[1:] *= np.cumprod(signs)[:, None]
    return q


def reflect_quat(q, axes):
    """Quaternion conjugation by an orthogonal diagonal coordinate change."""
    axes = np.asarray(axes)
    return np.asarray(q) * np.r_[np.prod(axes) * axes, 1.]


def euler_unity(degrees):
    """Unity Quaternion.Euler: extrinsic Z then X then Y."""
    a = np.deg2rad(np.asarray(degrees)) * .5
    x, y, z = np.moveaxis(a, -1, 0)
    zero = np.zeros_like(x)
    qx = np.stack((np.sin(x), zero, zero, np.cos(x)), axis=-1)
    qy = np.stack((zero, np.sin(y), zero, np.cos(y)), axis=-1)
    qz = np.stack((zero, zero, np.sin(z), np.cos(z)), axis=-1)
    return mul(mul(qy, qx), qz)


def quat_to_euler_unity(q):
    """Extract Unity extrinsic Z-X-Y (intrinsic Y-X-Z) Euler degrees from XYZW quaternion(s)."""
    q = unit(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    r12 = 2.0 * (y * z - x * w)
    ex = np.arcsin(np.clip(-r12, -1.0, 1.0))
    gimbal = np.abs(np.cos(ex)) < 1e-6
    ey = np.where(gimbal, np.arctan2(-2.0 * (x * z - y * w), 1.0 - 2.0 * (y * y + z * z)),
                  np.arctan2(2.0 * (x * z + y * w), 1.0 - 2.0 * (x * x + y * y)))
    ez = np.where(gimbal, 0.0,
                  np.arctan2(2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z)))
    return np.rad2deg(np.stack((ex, ey, ez), axis=-1))


def slerp(q0, q1, alpha):
    """Shortest-arc spherical linear interpolation for unit XYZW quaternions."""
    q0, q1 = unit(q0), unit(q1)
    alpha = np.asarray(alpha, dtype=float)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    alpha_e = alpha[..., None]
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    near = sin_t < 1e-6
    safe_sin = np.where(near, 1.0, sin_t)
    w0 = np.where(near, 1.0 - alpha_e, np.sin((1.0 - alpha_e) * theta) / safe_sin)
    w1 = np.where(near, alpha_e, np.sin(alpha_e * theta) / safe_sin)
    return unit(w0 * q0 + w1 * q1)


def world_pose(nodes, samples=None, indices=None):
    """Compose TRS hierarchy, retaining scales (FBX parents may be 100x)."""
    frames = 1 if samples is None else len(samples)
    count = len(nodes)
    p = np.empty((frames, count, 3)); q = np.empty((frames, count, 4))
    s = np.empty((count, 3))
    keyed = {} if indices is None else {n: i for i, n in enumerate(indices)}
    for i, node in enumerate(nodes):
        parent = node['parent']
        if parent >= i or parent < -1:
            raise ValueError("Backend node hierarchy must be parent-first")
        column = keyed.get(i)
        lp = np.asarray(node['position']) if column is None else samples[:, column, :3]
        lq = unit(node['rotation']) if column is None else unit(samples[:, column, 3:])
        ls = np.asarray(node['scale'])
        if parent < 0:
            p[:, i] = lp; q[:, i] = lq; s[i] = ls
        else:
            p[:, i] = p[:, parent] + rotate(q[:, parent], lp * s[parent])
            q[:, i] = unit(mul(q[:, parent], lq)); s[i] = s[parent] * ls
    return p, q
