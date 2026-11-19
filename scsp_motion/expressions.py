"""Deterministic source event baking; approximate envelopes are reported."""
from pathlib import Path
import collections
import re
import numpy as np
from .backend import read_json
from .vmd import stable_indices
from .math3d import unit,inv,rotate,between,continuous,mul


def mouth_for(slot,info,rules):
    route=rules['mouth_routes'].get(str(slot['slot']))
    if route:
        p=(Path(info['task_dir'])/route).resolve()
        if not p.is_relative_to(Path(info['task_dir']).resolve()) or not p.is_file():
            raise ValueError('自定义口型路径必须在任务目录内')
        return p
    alias=rules['character_aliases'].get(str(slot['character_id']))
    aliases=alias if isinstance(alias,list) else [alias] if alias else []
    candidates=[Path(p) for p in info['mouths'] if any(re.search(r'(?:^|_)'+re.escape(a)+r'_mouth$',Path(p).stem,re.I) for a in aliases)]
    if not candidates and slot.get('lip_sync'):
        lip=slot['lip_sync']
        candidates=[Path(p) for p in info['mouths'] if Path(p).stem.casefold()==f'{lip}_mouth'.casefold()]
    if len(candidates)!=1:
        raise ValueError(f"站位 {slot['slot']} 的角色 {slot['character_id']} 口型无法唯一配对；请配置 mouth_routes")
    return candidates[0]


def bake_morphs(exp_path,mouth_path,frame_count,rules,model,morph_names):
    times=np.arange(frame_count)/30; fnums=np.arange(frame_count,dtype=np.uint32)
    data=read_json(exp_path); events=data.get('events')
    if not isinstance(events,list):raise ValueError('表情 JSON 没有 events 数组')
    outputs={}; ignored=collections.Counter(); beyond=0
    events=sorted(enumerate(events),key=lambda x:(float(x[1]['startFrame']),x[0]))
    eye_names=['eye_L_close','eye_R_close','eye_L_smile','eye_R_smile']
    eye=np.zeros((frame_count,4)); frame60=times*rules['expression_fps']
    # A new state transition interrupts the earlier transition at its exact
    # source time, including multiple events at one frame in source order.
    transitions={name:(0.,0.,0.,0.) for name in eye_names}
    def transition(name,start,duration,to):
        old=transitions.get(name,(0.,0.,0.,0.)); a,b,t,d=old
        current=b if d<=0 else a+(b-a)*np.clip((start-t)/d,0,1)
        transitions[name]=(current,to,start,duration)
        values=current+(to-current)*np.clip((frame60-start)/max(duration,1e-12),0,1)
        if duration<=0:values=np.full(frame_count,to)
        target=outputs.setdefault(name,np.zeros(frame_count))
        mask=frame60>=start;target[mask]=values[mask]
    pulses=[]; gaze=[]
    for _,e in events:
        start=float(e['startFrame']); name=e['eventName']
        if not np.isfinite(start) or start<0:raise ValueError('表情事件帧号非法')
        if start>frame60[-1]+1e-4:beyond+=1;continue
        known=(name in ('CloseEye','CloseEyeSmile','OpenEye','OpenEyeSmile','Blink','ChangeTarget',
                         'MouthNormal','MouthLipSerious','MouthAnger','MouthShy','MouthSurprise')
               or name.startswith(('Wink','Eyebrow')))
        if not known:ignored[name]+=1;continue
        arg1=float(e.get('arg1') or 0) if name!='ChangeTarget' else 0
        arg2=float(e.get('arg2') or 0)
        if not np.isfinite([arg1,arg2]).all():raise ValueError('表情事件参数非有限值')
        if name in ('CloseEye','CloseEyeSmile','OpenEye','OpenEyeSmile'):
            for j,n in enumerate(eye_names):
                weight=np.clip(arg2/100,0,1) if ((name=='CloseEye' and j<2) or (name=='CloseEyeSmile' and j>=2)) else 0
                transition(n,start,max(0,arg1),weight)
        elif name.startswith('Wink') or name=='Blink':pulses.append((name,start,max(2,arg1),np.clip(arg2/100,0,1)))
        elif name.startswith('Eyebrow'):
            suffix={'EyebrowAddUp':'up','EyebrowAddDown':'down','EyebrowSad':'sad',
                    'EyebrowSerious':'serious','EyebrowUnique':'unique'}.get(name)
            if suffix:
                # Source eyebrow arg1 is amount, arg2 is blend duration.
                if suffix in ('sad','serious','unique'):
                    for other in ('sad','serious','unique'):
                        for side in ('L','R'):transition(f'eyebrow_{side}_{other}',start,max(0,arg2),0)
                for side in ('L','R'):transition(f'eyebrow_{side}_{suffix}',start,max(0,arg2),np.clip(arg1/100,0,1))
            elif name=='EyebrowNormal':
                for suffix in ('sad','serious','unique','up','down'):
                    for side in ('L','R'):transition(f'eyebrow_{side}_{suffix}',start,max(0,arg2),0)
            else:ignored[name]+=1
        elif name in ('MouthNormal','MouthLipSerious','MouthAnger','MouthShy','MouthSurprise'):
            sources=['mouth_serious','mouth_anger','mouth_shy','mouth_surp']
            chosen={'MouthLipSerious':'mouth_serious','MouthAnger':'mouth_anger','MouthShy':'mouth_shy','MouthSurprise':'mouth_surp'}.get(name)
            for n in sources:transition(n,start,max(0,arg2),np.clip(arg1/100,0,1) if n==chosen else 0)
        elif name=='ChangeTarget':gaze.append(e)
        else:ignored[name]+=1
    for name,start,duration,weight in pulses:
        elapsed=frame60-start
        if name=='Blink':amount=1-np.abs(2*np.clip(elapsed/duration,0,1)-1)
        else:
            edge=min(rules['wink_edge_frames'],duration/2)
            amount=np.minimum(np.clip(elapsed/edge,0,1),np.clip((duration-elapsed)/edge,0,1))
        for j,n in enumerate(eye_names):
            side=j%2;smile=j>=2
            if smile!=name.endswith('Smile'):continue
            if name.startswith('WinkL') and side!=0:continue
            if name.startswith('WinkR') and side!=1:continue
            outputs[n]=np.maximum(outputs.get(n,np.zeros(frame_count)),amount*weight)
    mouth=read_json(mouth_path); rate=float(mouth['frameRate']); samples=mouth['datas']
    if not 1<=rate<=240 or not samples:raise ValueError('口型采样率/数组非法')
    mt=times*rate+float(rules['mouth_offset_frames'])
    for vowel in 'aiueo':
        vals=np.asarray([f.get(vowel,0) for f in samples],dtype=float)
        if not np.isfinite(vals).all():raise ValueError('口型包含非有限权重')
        outputs[vowel.upper()]=np.interp(mt,np.arange(len(vals)),np.clip(vals,0,1),left=0,right=0)
    mapping=dict(morph_names['source_names']);mapping.update(rules['morph_map'])
    valid={m.name for m in model.morphs};tracks={};missing=[]
    for name,values in outputs.items():
        target=mapping.get(name,name)
        if target is None:continue
        if target not in valid:
            if np.any(values>1e-7):missing.append(target)
            continue
        if target in tracks:raise ValueError('多个源表情映射到相同形态键: '+target)
        keep=stable_indices(values[:,None]);tracks[target]=(fnums[keep],values[keep])
    if missing and not rules['allow_missing_morphs']:
        raise ValueError('目标 PMX 缺少有动作的形态键: '+', '.join(missing))
    return tracks,dict(source_offset_frame=mouth.get('offsetFrame'),applied_offset_frames=rules['mouth_offset_frames'],
                       ignored_events=dict(ignored),events_beyond_clip=beyond,missing_morphs=missing,
                       expression_policy='linear state transitions; triangular blink/trapezoid wink; eyebrow/mouth envelopes are compatibility approximations',
                       mouth_policy='sampled vowels at source rate; out-of-range zero; tongue/db not mapped'),gaze


def _yaw_pitch_to_quat(yaw_deg, pitch_deg):
    """Build local MMD eye quaternion from +X=left yaw and +Y=up pitch (forward=-Z)."""
    hy = np.deg2rad(-yaw_deg) * 0.5
    hp = np.deg2rad(pitch_deg) * 0.5
    sy, cy = np.sin(hy), np.cos(hy)
    sp, cp = np.sin(hp), np.cos(hp)
    # q_yaw(0, sy, 0, cy) * q_pitch(sp, 0, 0, cp)
    return np.stack((cy * sp, sy * cp, -sy * sp, cy * cp), axis=-1)


def _aim_world_target(eye_world_pos, mid_eye_pos, head_rot, target_world_pos,
                      max_yaw_deg, max_pitch_up_deg, max_pitch_down_deg,
                      min_conv_dist, fade_start_deg=75.0, fade_end_deg=115.0):
    """Aim from eye_world_pos toward target_world_pos in Head-local space with guards."""
    rel_mid = target_world_pos - mid_eye_pos
    dist_mid = np.linalg.norm(rel_mid, axis=-1, keepdims=True)
    safe_dir = np.where(dist_mid > 1e-6, rel_mid / np.maximum(dist_mid, 1e-6), np.array([0., 0., -1.]))
    safe_target = np.where(dist_mid < min_conv_dist, mid_eye_pos + safe_dir * min_conv_dist, target_world_pos)
    d = unit(rotate(inv(head_rot), safe_target - eye_world_pos))
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    fwd = -dz
    theta_deg = np.rad2deg(np.arctan2(np.hypot(dx, dy), fwd))
    fade = np.clip((fade_end_deg - theta_deg) / max(fade_end_deg - fade_start_deg, 1e-6), 0.0, 1.0)
    raw_yaw = np.rad2deg(np.arctan2(dx, np.maximum(fwd, 1e-4)))
    raw_pitch = np.rad2deg(np.arctan2(dy, np.maximum(np.hypot(dx, np.maximum(fwd, 0.0)), 1e-4)))
    clamped = int(np.sum(
        (np.abs(raw_yaw) > max_yaw_deg)
        | (raw_pitch > max_pitch_up_deg)
        | (raw_pitch < -max_pitch_down_deg)
        | (theta_deg > fade_start_deg)
    ))
    yaw = max_yaw_deg * np.tanh(raw_yaw / 30.0) * fade
    pitch_lim = np.where(raw_pitch >= 0.0, max_pitch_up_deg, max_pitch_down_deg)
    pitch = pitch_lim * np.tanh(raw_pitch / 25.0) * fade
    return yaw, pitch, clamped


def bake_gaze(events,model,world_positions,bone_tracks,all_actors,camera_positions,rules):
    """Bake bounded eye aiming in Head-local space. Symbolic targets are explicit approximations."""
    mode=rules.get('gaze_mode','source')
    if mode=='source':
        return {},dict(events=len(events),mode=mode,applied_event_gaze=False,
                       deferred_targets=dict(collections.Counter(str(e.get('arg1') or '') for e in events)),
                       policy='preserve source Humanoid eye bone motion; ChangeTarget events are retained but not reconstructed')
    if mode!='approximate':raise ValueError('gaze_mode 必须是 source 或 approximate')
    if not events:return {},dict(events=0,mode=mode)
    names={b.name:i for i,b in enumerate(model.bones)}
    head_idx=names['頭']
    count=len(world_positions[head_idx]);fps60=float(rules['expression_fps'])
    times=np.arange(count)/30.0*fps60
    max_yaw_deg=float(rules.get('gaze_max_yaw_deg',5.5))
    max_pitch_up_deg=float(rules.get('gaze_max_pitch_up_deg',3.0))
    max_pitch_down_deg=float(rules.get('gaze_max_pitch_down_deg',3.5))
    default_neg_frames=6.0
    min_conv_dist=1.5*float(rules['units_per_meter'])
    # Head-relative angular offsets (yaw +X=left deg, pitch +Y=up deg) so
    # directional cues rotate naturally with the character's head.
    dir_presets={
        'CenterUp':(0.0,max_pitch_up_deg),
        'CenterMiddle':(0.0,0.0),
        'CenterDown':(0.0,-max_pitch_down_deg),
        'LeftUp':(max_yaw_deg,max_pitch_up_deg),
        'LeftMiddle':(max_yaw_deg,0.0),
        'LeftDown':(max_yaw_deg,-max_pitch_down_deg),
        'RightUp':(-max_yaw_deg,max_pitch_up_deg),
        'RightMiddle':(-max_yaw_deg,0.0),
        'RightDown':(-max_yaw_deg,-max_pitch_down_deg),
        'HalfLeftUp':(0.5*max_yaw_deg,0.6*max_pitch_up_deg),
        'HalfLeftMiddle':(0.5*max_yaw_deg,0.0),
        'HalfLeftDown':(0.5*max_yaw_deg,-0.6*max_pitch_down_deg),
        'HalfRightUp':(-0.5*max_yaw_deg,0.6*max_pitch_up_deg),
        'HalfRightMiddle':(-0.5*max_yaw_deg,0.0),
        'HalfRightDown':(-0.5*max_yaw_deg,-0.6*max_pitch_down_deg),
        'BackUp':(0.0,0.5*max_pitch_up_deg),
        'BackMiddle':(0.0,0.0),
        'BackDown':(0.0,-0.5*max_pitch_down_deg),
    }
    world_choices={}
    if camera_positions is not None:world_choices['Camera']=np.asarray(camera_positions,dtype=float)
    for source,target in [('LeftHand','左手首'),('RightHand','右手首')]:
        if target in names:world_choices[source]=np.asarray(world_positions[names[target]],dtype=float)
    for slot,pos in all_actors.items():world_choices[f'Character{slot:02d}']=np.asarray(pos,dtype=float)
    for name,target in rules['eye_targets'].items():world_choices[name]=np.tile(target,(count,1))

    # Evaluate head global deformation from saved local rotations through the
    # ancestor chain. All inherited controls on this path are identity.
    from .retarget import topology
    identity=np.tile([0.,0,0,1.],(count,1)); global_rot={}
    for i in topology(model):
        b=model.bones[i];parent=identity if b.parent<0 else global_rot[b.parent]
        local=identity
        if b.name in bone_tracks:
            fs,vs=bone_tracks[b.name]
            # Sparse reduction only removes constant interiors.
            index=np.clip(np.searchsorted(fs,np.arange(count),side='right')-1,0,len(fs)-1)
            local=vs[index,3:]
        global_rot[i]=mul(parent,local)
    head_rot=global_rot[head_idx]
    eye_indices=[names[n] for n in ('左目','右目') if n in names]
    mid_eye_pos=(0.5*(world_positions[eye_indices[0]]+world_positions[eye_indices[1]])
                 if len(eye_indices)==2 else
                 world_positions[eye_indices[0]] if eye_indices else world_positions[head_idx])

    ordered_events=sorted(enumerate(events),key=lambda x:(float(x[1]['startFrame']),x[0]))
    unknown=collections.Counter()
    for _,e in ordered_events:
        tname=str(e.get('arg1') or '')
        if tname not in dir_presets and tname not in world_choices:
            unknown[tname]+=1

    result={};frames=np.arange(count,dtype=np.uint32);clamped=0
    for name in ('左目','右目'):
        if name not in names:continue
        eye_idx=names[name]
        eye_pos=world_positions[eye_idx]
        aim_cache={}
        for wname,wpos in world_choices.items():
            ty,tp,c=_aim_world_target(
                eye_pos,mid_eye_pos,head_rot,wpos,
                max_yaw_deg,max_pitch_up_deg,max_pitch_down_deg,min_conv_dist,
            )
            aim_cache[wname]=(ty,tp)
            clamped+=c
        cur_yaw=np.zeros(count,dtype=float)
        cur_pitch=np.zeros(count,dtype=float)
        for _,e in ordered_events:
            tname=str(e.get('arg1') or '')
            start=float(e['startFrame'])
            raw_arg2=float(e.get('arg2')) if e.get('arg2') not in (None,'') else -1.0
            duration=default_neg_frames if raw_arg2<0 else raw_arg2
            if tname in dir_presets:
                tgt_y,tgt_p=dir_presets[tname]
                tgt_y=np.full(count,tgt_y,dtype=float)
                tgt_p=np.full(count,tgt_p,dtype=float)
            elif tname in aim_cache:
                tgt_y,tgt_p=aim_cache[tname]
            else:
                # Unresolved stage target (e.g. Target2) safely returns gaze to neutral.
                tgt_y=np.zeros(count,dtype=float);tgt_p=np.zeros(count,dtype=float)
            mask=times>=start
            if not np.any(mask):continue
            idx0=min(int(round(start*30.0/fps60)),count-1)
            prev_y=cur_yaw[idx0];prev_p=cur_pitch[idx0]
            if duration<=0:
                alpha=np.ones(count,dtype=float)
            else:
                u=np.clip((times-start)/duration,0.0,1.0)
                alpha=u*u*(3.0-2.0*u)
            cur_yaw[mask]=(prev_y+(tgt_y-prev_y)*alpha)[mask]
            cur_pitch[mask]=(prev_p+(tgt_p-prev_p)*alpha)[mask]
        q=continuous(_yaw_pitch_to_quat(cur_yaw,cur_pitch))
        values=np.c_[np.zeros((count,3)),q]
        keep=stable_indices(values);result[name]=(frames[keep],values[keep])
    return result,dict(events=len(events),mode=mode,unknown_targets=dict(unknown),clamped_eye_frames=clamped,
                       policy='head-relative bounded eye aiming with behind-head fade, near-distance convergence guard, and smoothstep ChangeTarget transitions')
