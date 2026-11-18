"""Shared song clock with native-rate clips and explicit end-pose holds."""
import math
import numpy as np


def frame_count(duration, fps=30):
    """Include t=0 and the endpoint; ignore Unity float noise at whole frames."""
    if not math.isfinite(duration) or not 0 < duration <= 600:
        raise ValueError('动作时长须在 0 至 600 秒之间')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('无效采样率')
    last = duration * fps
    nearest = round(last)
    if abs(last - nearest) <= .001:
        last = nearest
    return math.ceil(last) + 1


def plan_song_timeline(slots, roots, fps=30, policy='hold'):
    if policy not in ('hold', 'strict'):
        raise ValueError('duration_policy 必须是 hold 或 strict')
    if not slots:
        raise ValueError('没有可转换的站位')
    tracks = {}
    for slot in slots:
        number = slot['slot']; body = float(slot['duration'])
        body_count = frame_count(body, fps)
        ends = [float(roots[number]['pos_' + axis]['m_Curve'][-1]['time']) for axis in 'xz']
        for end in ends:
            frame_count(end, fps)
            if abs(end - body) > .05:
                raise ValueError(f'站位 {number}: 身体动作 {body:.5f}s 与本人的根位移 {end:.5f}s 时长不一致')
        tracks[str(number)] = dict(body_duration=body, root_duration=max(ends),
                                   source_frame_count=body_count)
    if policy == 'strict' and len({t['source_frame_count'] for t in tracks.values()}) > 1:
        raise ValueError('多人主动作时长不一致；duration_policy=strict 要求各站位等长')
    duration = max(max(t['body_duration'], t['root_duration']) for t in tracks.values())
    count = frame_count(duration, fps)
    for t in tracks.values():
        t.update(body_hold_tail_seconds=max(0., duration-t['body_duration']),
                 root_hold_tail_seconds=max(0., duration-t['root_duration']))
    return dict(policy=policy, fps=fps, duration=duration, frame_count=count,
                last_vmd_frame=count-1, sampled_end_seconds=(count-1)/fps,
                slots=tracks, tail_policy='hold last body pose and each root curve endpoint; expression/mouth retain song time')


def hold_pose_tail(positions, rotations, count):
    """Keep all existing sample times and repeat the final pose only at the end."""
    if (positions.ndim != 3 or rotations.shape != positions.shape[:2]+(4,)
            or positions.shape[2] != 3 or not len(positions)):
        raise ValueError('骨骼姿态数组形状无效')
    if len(positions) > count:
        raise ValueError('实际动作帧数超过歌曲时间轴，不能截断源动作')
    extra = count-len(positions)
    if not extra:
        return positions, rotations
    return (np.concatenate((positions, np.repeat(positions[-1:], extra, axis=0))),
            np.concatenate((rotations, np.repeat(rotations[-1:], extra, axis=0))))
