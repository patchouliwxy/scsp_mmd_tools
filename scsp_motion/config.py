from pathlib import Path
import copy
import math
import sys
from .backend import read_json


ROOT=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
if not (ROOT/'Resources').is_dir():ROOT=Path(sys.prefix)


def _valid_offset_vec(v):
    return (isinstance(v,(list,tuple)) and len(v)==3
            and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in v))


def load_rules(path=None):
    rules=read_json(ROOT/'Resources/default-rules.json')
    if path:
        override=read_json(path)
        if override.get('schema',1)!=1:raise ValueError('未知自定义规则版本')
        unknown=set(override)-set(rules)
        if unknown:raise ValueError('未知自定义规则字段: '+', '.join(sorted(unknown)))
        for k,v in override.items():
            if k=='camera_shooting_offsets' and (v is None or v=={}):
                rules[k]={}
            elif isinstance(v,dict) and isinstance(rules.get(k),dict):
                rules[k].update(copy.deepcopy(v))
            else:
                rules[k]=copy.deepcopy(v)
    if rules['fps']!=30:raise ValueError('标准 MMD VMD 固定 30 FPS；不能将源 60 FPS 帧号直接写入')
    if rules['ik_mode'] not in ('fk','ik'):raise ValueError('ik_mode 必须是 fk 或 ik')
    if rules['formation_origin'] not in ('group','world'):raise ValueError('formation_origin 必须是 group 或 world')
    if rules['camera_mode'] not in ('adapted','off'):raise ValueError('camera_mode 必须是 adapted 或 off')
    if rules['camera_interpolation'] not in ('smooth','legacy'):raise ValueError('camera_interpolation 必须是 smooth 或 legacy')
    if not isinstance(rules['camera_zoom_compensation'],bool):raise ValueError('camera_zoom_compensation 必须是 true 或 false')
    ms=rules['camera_zoom_compensation_max_shift_pmx']
    if ms is not None and (not isinstance(ms,(int,float)) or isinstance(ms,bool) or not math.isfinite(ms) or ms<0):
        raise ValueError('camera_zoom_compensation_max_shift_pmx 必须是 null 或非负有限数值')
    if not isinstance(rules['camera_sparse_reduce'],bool):raise ValueError('camera_sparse_reduce 必须是 true 或 false')
    if not isinstance(rules['camera_noise_filter'],bool):raise ValueError('camera_noise_filter 必须是 true 或 false')
    ths=rules['camera_target_height_scale']
    if not (isinstance(ths,bool) or ths=='auto'):raise ValueError('camera_target_height_scale 必须是 true、false 或 auto')
    if not isinstance(rules['camera_dynamic_actor_anchor'],bool):raise ValueError('camera_dynamic_actor_anchor 必须是 true 或 false')
    if not isinstance(rules['camera_shooting_offsets'],dict) or any(v is not None and not _valid_offset_vec(v) for v in rules['camera_shooting_offsets'].values()):
        raise ValueError('camera_shooting_offsets 的每个值必须是 null 或 3 个有限数值')
    if rules['gaze_mode'] not in ('source','approximate'):raise ValueError('gaze_mode 必须是 source 或 approximate')
    for gk in ('gaze_max_yaw_deg','gaze_max_pitch_up_deg','gaze_max_pitch_down_deg'):
        gv=rules[gk]
        if not isinstance(gv,(int,float)) or isinstance(gv,bool) or not math.isfinite(gv) or not 0<=gv<=30:
            raise ValueError(f'{gk} 必须是 0..30 之间的有限数值')
    if rules['duration_policy'] not in ('hold','strict'):raise ValueError('duration_policy 必须是 hold 或 strict')
    return rules
