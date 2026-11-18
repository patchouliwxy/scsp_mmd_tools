"""Source global deformation -> PMX global-axis local bone keys."""
import numpy as np
from . import pmx_io
from .config import ROOT
from .backend import read_json
from .math3d import unit,mul,inv,rotate,between,reflect_quat,continuous
from .vmd import stable_indices


def _validate_target(model):
    names=[b.name for b in model.bones]
    if len(names)!=len(set(names)):raise ValueError('目标 PMX 存在重复骨名')
    required=['全ての親','センター','下半身','上半身','左腕','右腕','左ひじ','右ひじ','左足','右足','左ひざ','右ひざ','左足首','右足首']
    missing=[n for n in required if n not in names]
    if missing:raise ValueError('目标 PMX 缺少必要骨: '+', '.join(missing))
    positions=np.asarray([b.location for b in model.bones])
    if not np.isfinite(positions).all():raise ValueError('目标 PMX 骨坐标非有限值')
    return model


def load_target(path):
    return _validate_target(pmx_io.load(path))


def load_generic_target(path=None):
    data=read_json(path or (ROOT/'Resources/generic_mmd_profile.json'))
    model=pmx_io.Model()
    model.name=data.get('name','SCSP Generic MMD')
    for row in data['bones']:
        bone=pmx_io.Bone()
        bone.name=row['name'];bone.parent=int(row['parent']);bone.location=list(row['location'])
        model.bones.append(bone)
    for name in data.get('morphs',[]):
        morph=pmx_io.VertexMorph();morph.name=name
        model.morphs.append(morph)
    return _validate_target(model)


def topology(model):
    order=[]; visiting=set(); done=set()
    def add(i):
        if i in visiting:raise ValueError('PMX 骨层级存在循环')
        if i in done:return
        visiting.add(i); parent=model.bones[i].parent
        if parent>=len(model.bones) or parent==i:raise ValueError('PMX 父骨索引无效')
        if parent>=0:add(parent)
        visiting.remove(i);done.add(i);order.append(i)
    for i in range(len(model.bones)):add(i)
    return order


def convert_pose(model,source_model,calibration,positions,rotations,root_motion,rules,mappings):
    frames=len(positions); fnums=np.arange(frames,dtype=np.uint32)
    source={n['name']:i for i,n in enumerate(source_model['nodes'])}
    target={b.name:i for i,b in enumerate(model.bones)}
    mapping={}
    for rule in mappings['mappings']:
        name=rule['japanese']; candidates=rule['source_names']+rule.get('source_metadata',[])
        # The source Avatar uses ForeArm; our PMX promotes Elbow to forearm.
        candidates=[n.replace('Elbow','ForeArm') for n in candidates]
        if name in target:
            found=[source[n] for n in candidates if n in source]
            if found:mapping[name]=found[0]
    custom_targets=set()
    for src,dst in rules['bone_map'].items():
        if dst is None:continue
        if src not in source or dst not in target:raise ValueError('自定义骨映射找不到源/目标: '+src+' -> '+str(dst))
        if dst in custom_targets:raise ValueError('自定义规则有重复目标骨: '+dst)
        custom_targets.add(dst)
        mapping={name:idx for name,idx in mapping.items() if idx!=source[src]}
        mapping[dst]=source[src]
    mapping.pop('全ての親',None)
    if len(set(mapping.values()))!=len(mapping):raise ValueError('一个源骨被映射到多个目标骨，请使用姿态规则而不是重复映射')
    axes=np.asarray(rules['source_axes'],dtype=float)
    if axes.shape!=(3,) or not np.all(np.isin(axes,[-1,1])):raise ValueError('source_axes 必须是三个 ±1')
    rest_p=np.asarray(calibration['neutral_positions'])*axes
    rest_q=reflect_quat(np.asarray(calibration['neutral_rotations']),axes)
    world_q=reflect_quat(rotations,axes)
    world_p=positions*axes
    target_p=np.asarray([b.location for b in model.bones],dtype=float)
    source_to_target={idx:target[name] for name,idx in mapping.items()}
    corrections={}; desired={}; correction_report={}; correction_references={}
    reverse={idx:name for name,idx in mapping.items()}
    preferred={'Hips':'Spine','Spine':'Spine1','Spine1':'Spine2','Spine2':'Neck','Neck':'Head',
               'LeftShoulder':'LeftArm','RightShoulder':'RightArm',
               'LeftArm':'LeftForeArm','RightArm':'RightForeArm',
               'LeftForeArm':'LeftHand','RightForeArm':'RightHand',
               'LeftUpLeg':'LeftLeg','RightUpLeg':'RightLeg','LeftLeg':'LeftFoot','RightLeg':'RightFoot',
               'LeftFoot':'LeftToe','RightFoot':'RightToe','LeftHand':'LeftHandMiddle1','RightHand':'RightHandMiddle1'}
    for name,idx in mapping.items():
        srcname=source_model['nodes'][idx]['name']; child=source.get(preferred.get(srcname,''))
        # Head/eye pivots are independently repositioned by the model pipeline.
        # A Head -> EYE vector is not an anatomical head orientation reference:
        # choosing one eye introduced a constant ~50-degree head tilt.
        terminal_orientation=srcname in ('Head','EYE_L','EYE_R','p00__EYE_L','p00__EYE_R') or name in ('頭','左目','右目')
        if terminal_orientation:
            child=None
        elif child not in source_to_target:
            children=[j for j in source_to_target if source_model['nodes'][j]['parent']==idx]
            child=children[0] if children else None
        if child is not None:
            sv=rest_p[child]-rest_p[idx]; tv=target_p[source_to_target[child]]-target_p[target[name]]
            correction=between(sv,tv) if min(np.linalg.norm(sv),np.linalg.norm(tv))>1e-7 else np.array([0.,0,0,1])
            correction_references[name]=dict(method='anatomical_segment',source_child=source_model['nodes'][child]['name'])
        else:
            parent=source_model['nodes'][idx]['parent']
            while parent>=0 and parent not in corrections:
                parent=source_model['nodes'][parent]['parent']
            correction=corrections.get(parent,np.array([0.,0,0,1]))
            correction_references[name]=dict(method='inherited_rest_frame',source_ancestor=source_model['nodes'][parent]['name'] if parent>=0 else None)
        if name in rules['rest_corrections']:
            correction=unit(rules['rest_corrections'][name])
            correction_references[name]=dict(method='explicit_rule')
        corrections[idx]=correction
        delta=mul(world_q[:,idx],inv(rest_q[idx]))
        desired[target[name]]=continuous(mul(delta,inv(correction)))
        correction_report[name]=correction.tolist()
    order=topology(model); identity=np.tile([0.,0,0,1.],(frames,1))
    world_rot={}; world_pos={}; tracks={}; locals_={}
    hips=source['Hips']; hips_target=target['下半身']
    scale=target_p[hips_target,1]/(rest_p[hips,1]*rules['units_per_meter'])
    if rules['body_scale'] is not None:scale=float(rules['body_scale'])
    if not .2<scale<5:raise ValueError('源/目标体型比例异常')
    center=(world_p[:,hips]-rest_p[hips])*rules['units_per_meter']*scale
    root=target['全ての親']; center_idx=target['センター']
    for i in order:
        bone=model.bones[i]; parent=bone.parent
        pq=identity if parent<0 else world_rot[parent]
        if i in desired:
            local=continuous(mul(inv(pq),desired[i]))
        else:local=identity
        world_rot[i]=unit(mul(pq,local)); locals_[i]=local
        offset=target_p[i] if parent<0 else target_p[i]-target_p[parent]
        displacement=root_motion if i==root else center if i==center_idx else np.zeros((frames,3))
        world_pos[i]=(np.zeros((frames,3)) if parent<0 else world_pos[parent])+rotate(pq,offset+displacement)
        if i in desired or i in (root,center_idx):
            values=np.concatenate((displacement,local),axis=1)
            keep=stable_indices(values); tracks[bone.name]=(fnums[keep],values[keep])
    # Eye keys are deliberately dense, including neutral keys, so loading the
    # corrected motion can replace the old per-frame approximate eye keys.
    # The common controller is reset because both per-eye channels are authored.
    for eye in ('左目','右目'):
        if eye in target and target[eye] in desired:
            tracks[eye]=(fnums,np.c_[np.zeros((frames,3)),locals_[target[eye]]])
    if '両目' in target:
        tracks['両目']=(fnums,np.c_[np.zeros((frames,3)),identity])
    ik_entries={}
    for side in ('左','右'):
        control=target.get(side+'足ＩＫ'); tip=target.get(side+'つま先ＩＫ')
        if control is None:
            if rules['ik_mode']=='ik':raise ValueError('目标模型缺少足 IK')
            continue
        ankle=target[side+'足首']; toe=target.get(side+'つま先')
        ik_entries[side+'足ＩＫ']=rules['ik_mode']=='ik'
        if tip is not None:ik_entries[side+'つま先ＩＫ']=rules['ik_mode']=='ik'
        # Store editable IK targets in both modes; FK mode disables solver.
        parent=model.bones[control].parent; pq=world_rot[parent]
        pp=world_pos[parent]
        local_p=rotate(inv(pq),world_pos[ankle]-pp)-(target_p[control]-target_p[parent])
        local_q=continuous(mul(inv(pq),world_rot[ankle]))
        val=np.concatenate((local_p,local_q),axis=1);keep=stable_indices(val)
        tracks[side+'足ＩＫ']=(fnums[keep],val[keep])
        if toe is not None and tip is not None:
            local_p=rotate(inv(world_rot[ankle]),world_pos[toe]-world_pos[ankle])-(target_p[tip]-target_p[control])
            val=np.concatenate((local_p,identity),axis=1);keep=stable_indices(val)
            tracks[side+'つま先ＩＫ']=(fnums[keep],val[keep])
    return tracks,[(0,ik_entries)],dict(mapped_bones=list(mapping),body_scale=scale,rest_corrections=correction_report,
                                      rest_correction_references=correction_references,
                                      ik_mode=rules['ik_mode'],retarget_method='global deformation with target rest-direction correction',
                                      pelvis_min=center.min(axis=0).tolist(),pelvis_max=center.max(axis=0).tolist()),world_pos
