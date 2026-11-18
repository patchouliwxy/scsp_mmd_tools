"""SCSP custom shot samples to dense 30 fps VMD camera keys."""
import collections
import functools
import numpy as np
from .backend import read_json
from .config import ROOT
from .curves import sample
from .math3d import euler_unity,mul,quat_to_euler_unity,rotate
from .camera_smooth import sample_series,encode_camera,reduce_block_keys


@functools.lru_cache(maxsize=1)
def _character_height_scales():
    data=read_json(ROOT/'Resources/character_heights.json')
    return {int(r['id']):float(r['heightScale']) for r in data.get('characters',[])}


def _sample_bone(arr,frames_30):
    arr=np.asarray(arr,dtype=float)
    clipped=np.clip(frames_30,0.,len(arr)-1)
    i0=np.floor(clipped).astype(int)
    i1=np.minimum(i0+1,len(arr)-1)
    frac=(clipped-i0)[:,None]
    return (1.-frac)*arr[i0]+frac*arr[i1]


def xyz(value):return np.array([value.get(a,0) for a in 'xyz'],dtype=float)


def _merge_easing_duplicates(curve):
    """Exported easing may repeat a join key; retain its left/right tangents."""
    keys=[];removed=0
    for key in curve['m_Curve']:
        if keys and key['time']==keys[-1]['time']:
            previous=keys[-1]
            if key['value']!=previous['value']:
                raise ValueError('相机缓动同一时间存在不同值，无法确定曲线')
            merged=dict(previous)
            for field in ('outSlope','outWeight'):
                if field in key:merged[field]=key[field]
            merged['weightedMode']=(int(previous.get('weightedMode',0))&1)|(int(key.get('weightedMode',0))&2)
            keys[-1]=merged;removed+=1
        else:keys.append(dict(key))
    return dict(m_Curve=keys),removed


def _smooth_interior(values,passes=2):
    arr=np.asarray(values,dtype=float).copy()
    if len(arr)<5:return arr
    for _ in range(passes):
        arr[1:-1]=0.25*arr[:-2]+0.5*arr[1:-1]+0.25*arr[2:]
    return arr


def _shot_time_grid(easing,n):
    """Map raw camera sample indices onto physical time [0,1] when easing is monotonic."""
    ek=easing.get('m_Curve') or []
    if n<3 or len(ek)<2:return None
    e_times=np.array([float(k['time']) for k in ek])
    e_vals=np.array([float(k['value']) for k in ek])
    if e_times[0]>1e-5 or e_times[-1]>1.+1e-5 or e_vals[0]>1e-5 or e_vals[-1]<1.-1e-5:return None
    if np.any(np.diff(e_times)<=1e-9) or np.any(np.diff(e_vals)<=0):return None
    u_grid=np.linspace(0.,1.,n)
    if len(ek)==n and np.max(np.abs(e_vals-u_grid))<1e-4:
        return e_times
    t_fine=np.linspace(0.,min(1.,e_times[-1]),max(4096,32*n))
    u_fine=sample(easing,t_fine)
    if np.any(np.diff(u_fine)<-1e-4):return None
    u_mono=np.maximum.accumulate(u_fine)
    if u_mono[0]>1e-5 or u_mono[-1]<1.-1e-5:return None
    t_grid=np.interp(u_grid,u_mono,t_fine)
    if np.any(np.diff(t_grid)<=1e-9):return None
    return t_grid


def _shooting_offset_unity(shooting_offsets,stype,applied):
    raw_off=shooting_offsets.get(stype)
    shoot_off=np.asarray(raw_off,dtype=float) if raw_off is not None else np.zeros(3)
    if raw_off is not None and np.any(np.abs(shoot_off)>0):
        applied['shootingTypeOffset']+=1
    return shoot_off


def _actor_anchor_mmd(w,sample_frames_30,actor_positions):
    if not actor_positions or w.get('shootingType')=='Stage':return None
    objs=w.get('objectList') or []
    available=[int(s) for s in objs if int(s) in actor_positions]
    if not available:return None
    scalar=np.ndim(sample_frames_30)==0
    frames=np.atleast_1d(np.asarray(sample_frames_30,dtype=float))
    stype=w.get('shootingType','')
    preferred={
        'FaceUp':('首','頭','上半身3','上半身2','上半身','下半身'),
        'BustUp':('首','頭','上半身3','上半身2','上半身','下半身'),
        'UpperUp':('首','頭','上半身3','上半身2','上半身','下半身'),
        'LowerUp':('下半身','腰','センター','上半身'),
        'FullBody':('下半身','腰','センター','上半身'),
        'RootBase':('腰','下半身','センター'),
        'LeftHand':('左手首','左ひじ','首','上半身3','下半身'),
        'RightHand':('右手首','右ひじ','首','上半身3','下半身'),
    }.get(stype, ('首','上半身3','上半身2','上半身','頭','下半身'))
    pts=[]
    for s in available:
        bones=actor_positions[s]
        bname=next((b for b in preferred if b in bones),None)
        if bname is None:continue
        pts.append(_sample_bone(bones[bname],frames))
    if not pts:return None
    mean_pts=np.mean(pts,axis=0)
    return mean_pts[0] if scalar else mean_pts


def bake_camera(path,count,origin,root_basis,rules,actor_positions=None,slot_characters=None):
    smooth=rules.get('camera_interpolation','smooth')=='smooth'
    data=read_json(path); shots=sorted(data['cameraWorkData'],key=lambda w:w['startFrame'])
    moves={}
    for m in data['customCameraMoveData']:
        if m['moveID'] in moves:raise ValueError('相机 moveID 重复')
        moves[m['moveID']]=m
    sample_count=(count-1)*2+1 if smooth else count
    positions=np.zeros((sample_count,3)); rotations=np.zeros((sample_count,3)); fovs=np.full(sample_count,30.)
    zoom_refs=np.zeros((sample_count,3))
    source_times=np.arange(sample_count)/(60 if smooth else 30); ignored=collections.Counter(); applied=collections.Counter(); segments=[]
    camera_axes=np.asarray(rules['camera_axes']);scale=float(rules['units_per_meter'])
    height_scales=_character_height_scales()
    slot_map={int(k):int(v) for k,v in (slot_characters or {}).items()}
    use_height_scale=bool(rules.get('camera_target_height_scale',True))
    shooting_offsets=rules.get('camera_shooting_offsets') or {}
    dynamic_anchor=bool(rules.get('camera_dynamic_actor_anchor',True))
    noise_filter=bool(rules.get('camera_noise_filter',False))
    sparse_reduce=bool(rules.get('camera_sparse_reduce',False))
    max_shift_pmx=rules.get('camera_zoom_compensation_max_shift_pmx')
    if camera_axes.shape!=(3,) or not np.all(np.isin(camera_axes,[-1,1])):raise ValueError('camera_axes 必须是 ±1')
    covered=np.zeros(sample_count,dtype=bool)
    for j,w in enumerate(shots):
        rate=float(w.get('frameRate',60));start=float(w['startFrame'])/rate
        if start>source_times[-1]+1e-6:continue
        end=float(shots[j+1]['startFrame'])/float(shots[j+1].get('frameRate',60)) if j+1<len(shots) else source_times[-1]+1/30
        m=moves.get(w['moveID'])
        if not m or not m.get('cameraPos'):
            raise ValueError('有效片段缺少相机采样数据: '+w['moveID'])
        n=len(m['cameraPos'])
        if len(m['tiltPanRoll'])!=n or len(m['zoom'])!=n:raise ValueError('相机采样数组长度不一致')
        mask=(source_times>=start-1e-8)&(source_times<end-1e-8)
        if not mask.any():continue
        t_raw=np.clip((source_times[mask]-start)/max(end-start,1e-8),0,1)
        speed=float(w.get('moveSpeed',1))
        if not np.isfinite(speed) or speed<=0:raise ValueError('相机 moveSpeed 必须为正有限数')
        t_prog=np.power(t_raw,speed) if abs(speed-1.)>=1e-6 else t_raw
        t=t_prog;easing_curve=None
        if w.get('easing',{}).get('m_Curve'):
            easing_curve,removed=_merge_easing_duplicates(w['easing'])
            if removed:applied['easing_duplicate_keys_merged']+=removed
            t=np.clip(sample(easing_curve,t_prog),0,1)
        raw_pos=np.array([xyz(v) for v in m['cameraPos']])
        raw_rot=np.unwrap(np.deg2rad([xyz(v) for v in m['tiltPanRoll']]),axis=0)
        raw_zoom=np.asarray(m['zoom'],dtype=float)
        if noise_filter and n>=5:
            raw_pos=_smooth_interior(raw_pos)
            raw_rot=_smooth_interior(raw_rot)
            raw_zoom=_smooth_interior(raw_zoom)
            applied['noiseFilter']+=1
        if smooth:
            t_grid=_shot_time_grid(easing_curve,n) if easing_curve is not None else None
            if t_grid is not None:
                pos=sample_series(raw_pos,t_prog,grid_x=t_grid)
                rot=sample_series(raw_rot,t_prog,grid_x=t_grid)
                zoom=sample_series(raw_zoom,t_prog,grid_x=t_grid)
            else:
                pos=sample_series(raw_pos,t);rot=sample_series(raw_rot,t);zoom=sample_series(raw_zoom,t)
        else:
            grid=np.linspace(0,1,n)
            pos=np.stack([np.interp(t,grid,raw_pos[:,a]) for a in range(3)],axis=1)
            rot=np.stack([np.interp(t,grid,raw_rot[:,a]) for a in range(3)],axis=1)
            zoom=np.interp(t,grid,raw_zoom)
        off_rot=xyz(w.get('offsetRotate',{}))
        if np.any(np.abs(off_rot)>1e-6):
            q_off=euler_unity(off_rot)
            pos=rotate(q_off,pos)
            rot=np.unwrap(np.deg2rad(quat_to_euler_unity(mul(q_off,euler_unity(np.rad2deg(rot))))),axis=0)
            applied['offsetRotate']+=1
        if rules['camera_fov_mode']=='focal_mm':zoom=np.rad2deg(2*np.arctan(24/(2*zoom)))
        elif rules['camera_fov_mode']!='degrees':raise ValueError('camera_fov_mode 必须是 degrees 或 focal_mm')
        if not np.isfinite(pos).all() or not np.isfinite(rot).all() or not np.isfinite(zoom).all():raise ValueError('相机数据非有限值')
        # SCSP body/root coordinates and shot camera coordinates use distinct
        # bases. Per-character shots supply a source target snapshot explicitly,
        # or fall back to the solved character anchor when useStartTargetData=false.
        indices=np.flatnonzero(mask)
        start_vmd=int(np.ceil(indices[0]/(2 if smooth else 1)))
        mid_offset=1 if (smooth and indices[0]%2 and len(indices)>1) else 0
        stype=w.get('shootingType','')
        if w.get('useStartTargetData'):
            base=xyz(w.get('targetBasePoint',{}));norm=xyz(w.get('targetNormalized',{}))
            h_scale=1.
            if use_height_scale and slot_map:
                objs=w.get('objectList') or []
                hs=[height_scales[slot_map[int(s)]]
                    for s in objs if int(s) in slot_map and slot_map[int(s)] in height_scales]
                if hs:
                    h_scale=float(np.mean(hs))
                    applied['targetHeightScale']+=1
            shoot_off=_shooting_offset_unity(shooting_offsets,stype,applied)
            v=base+h_scale*norm+shoot_off
            anchor=np.array([v[0],v[2]])@root_basis
            anchor[1]+=v[1]
            shot_pos=((pos*camera_axes+anchor)-origin)*scale
            zoom_reference=(anchor-origin)*scale
            shot_zoom_refs=np.broadcast_to(zoom_reference,(len(indices),3))
        else:
            move_trace=w.get('moveTrace')
            use_dynamic=dynamic_anchor and (move_trace is None or float(move_trace)>=0.)
            start_actor_frame=float(np.clip(start*30.,0.,max(count-1,0)))
            actor_frames=(source_times[mask]*30.) if use_dynamic else np.full(len(indices),start_actor_frame)
            fallback_seq=_actor_anchor_mmd(w,actor_frames,actor_positions)
            if fallback_seq is not None:
                shoot_off=_shooting_offset_unity(shooting_offsets,stype,applied)
                anchor_off=np.array([shoot_off[0],shoot_off[2]])@root_basis
                anchor_off[1]+=shoot_off[1]
                fallback_seq=fallback_seq+anchor_off*scale
                shot_pos=(pos*camera_axes)*scale+fallback_seq
                shot_zoom_refs=fallback_seq
                zoom_reference=fallback_seq[mid_offset]
                applied['objectListAnchorFallback']+=1
                if use_dynamic:applied['dynamicActorAnchor']+=1
            else:
                if stype!='Stage' and w.get('objectList'):
                    ignored['missing_actor_anchor']+=1
                anchor=xyz(dict(zip('xyz',rules['camera_stage_offset'])))
                shot_pos=((pos*camera_axes+anchor)-origin)*scale
                stage_seq=_actor_anchor_mmd(dict(shootingType='BustUp',objectList=list(actor_positions or {})),actor_frames,actor_positions)
                if stage_seq is not None:
                    shot_zoom_refs=stage_seq
                    zoom_reference=stage_seq[mid_offset]
                else:
                    zoom_reference=np.array([0.,scale,0.])
                    shot_zoom_refs=np.broadcast_to(zoom_reference,(len(indices),3))
        shot_rot=-rot*camera_axes
        shot_fov=np.clip(zoom,1,179)
        trans=float(w.get('moveTransition') or 0)
        if trans>1e-6:
            if smooth:applied['moveTransition_approximate']+=1
            else:ignored['moveTransition']+=1
        positions[mask]=shot_pos
        rotations[mask]=shot_rot
        zoom_refs[mask]=shot_zoom_refs
        fovs[mask]=shot_fov;covered[mask]=True
        for feature in ('flip','swayReset'):
            value=w.get(feature)
            nonzero=any(abs(x)>1e-6 for x in value.values()) if isinstance(value,dict) else bool(value)
            if nonzero:ignored[feature]+=1
        if w.get('moveTrace',0) not in (0,-1):ignored['moveTrace']+=1
        if int(indices[-1]//(2 if smooth else 1))<start_vmd:
            ignored['subframe_only_shot']+=1
            continue
        segments.append(dict(move_id=w['moveID'],start_frame_60=w['startFrame'],start_vmd_frame=start_vmd,
                             end_vmd_frame=int(indices[-1]//(2 if smooth else 1)),samples=n,hard_cut=(start_vmd>0 and trans<=1e-6),
                             transition_seconds=max(0.,trans),zoom_reference=zoom_reference.tolist()))
    if not covered.all():raise ValueError('相机时间轴未完整覆盖主动作')
    if smooth:
        if sparse_reduce:applied['sparseReduce']+=1
        keys,optical_centers,smoothing=encode_camera(positions,rotations,fovs,segments,
            compensate=rules.get('camera_zoom_compensation',True),source_fps=60,zoom_references=zoom_refs,
            max_shift_pmx=max_shift_pmx,sparse_reduce=sparse_reduce)
        return keys,optical_centers,dict(mode='adapted',shots=segments,applied_features=dict(applied),
            unsupported_features=dict(ignored),smoothing=smoothing,
            policy='shape-preserving sample interpolation; dense cubic VMD camera; reference-plane fractional zoom compensation; preserve hard cuts',
            limitations=['缩放补偿保持参考深度的投影，其他景深存在小幅透视差异；极快或帧内反向曲线仍有 30 FPS 整帧表示残差；moveTransition 使用按秒的 smoothstep 兼容近似，未验证原游戏混合语义。',
                         'UCS 切段、部分舞台特例及 moveTrace/flip/swayReset 尚未还原；目标点 heightScale/shootingType 偏移与动态骨骼锚点已按实机标定，若目标模型体型与主数据差异较大仍需实际画面验收。'])
    # Zero distance stores the optical center directly; Euler components follow
    # VMD's camera sign convention. Reduce per-shot segments via cubic Bezier
    # interpolation so integer FOV ramps interpolate smoothly without stepping,
    # while hard cuts use step interpolation on the incoming shot keyframe.
    rotations=rotations.copy()
    keys=[]
    linear_interp=bytes([20,107,20,107]*6)
    step_interp=bytes([127,127,0,0]*6)
    for seg in segments:
        a,b=seg['start_vmd_frame'],seg['end_vmd_frame']
        rotations[a:b+1]=np.unwrap(rotations[a:b+1],axis=0)
        if a==0 or seg['hard_cut']:
            rotations[a:b+1]-=2*np.pi*np.round(rotations[a]/(2*np.pi))
        else:
            rotations[a:b+1]+=2*np.pi*np.round((rotations[a-1]-rotations[a])/(2*np.pi))
        in_interp=step_interp if seg['hard_cut'] else linear_interp
        keys.extend(reduce_block_keys(positions,rotations,fovs,a,b,in_interp))
    return keys,positions,dict(mode='adapted',shots=segments,applied_features=dict(applied),
                               unsupported_features=dict(ignored),
                               policy='sampled shot motion; target snapshots (with objectList bone anchor fallback); offsetRotate; per-shot VMD Bezier reduction with step hard-cuts; zero-distance optical center; FOV interpreted by configured mode',
                               limitations=['角色动态跟随（如实时目标追踪）、部分舞台特例相机及 UCS 切段未完整还原；需实际画面验收。'])
