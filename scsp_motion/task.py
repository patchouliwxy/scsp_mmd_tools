"""Read-only downloader export discovery using source-backed report identities."""
import functools
from pathlib import Path
import re
from .backend import read_json,sha
from .config import ROOT


@functools.lru_cache(maxsize=1)
def _builtin_scenecreate():
    path=ROOT/'Resources/song_scenecreate_slots.json'
    if not path.is_file():return {}
    return read_json(path).get('songs',{})


def _scenecreate_for_song(task,song):
    sc_file=task/'data'/f'{song}_scenecreate.json'
    if sc_file.is_file():
        ccd=read_json(sc_file).get('characterCreateData') or []
        anim_slots=[];lip_syncs=[]
        for idx,item in enumerate(ccd,1):
            anim=item.get('animName','')
            m=re.search(r'_(\d+)$',anim)
            anim_slots.append(int(m.group(1)) if m else idx)
            lip=item.get('lipSyncName','')
            lip_syncs.append(lip.split('/')[-1] if lip else '')
        return anim_slots,lip_syncs,'task_scenecreate'
    entry=_builtin_scenecreate().get(song.lower())
    if entry:
        return entry.get('anim_slots'),entry.get('lip_syncs'),'builtin_scenecreate'
    return None,None,'sequential'


def safe_file(root,value):
    path=(Path(root)/value).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('输入文件缺失或超出任务目录: '+str(value))
    return path


def inspect_task(task_dir):
    task=Path(task_dir).expanduser().resolve(strict=True)
    job=read_json(task/'job.json'); manifest=read_json(task/'manifest.json')
    if job.get('status')!='completed' or job.get('dependency_complete') is False or job.get('missing_count',0):
        raise ValueError('下载任务未完成或有缺失依赖')
    report=read_json(task/'animations/export-report.json')
    sources={Path(s['name']).name.casefold():s for s in report['sources']}
    selection=manifest.get('song_selection',{}); song=selection.get('code')
    if not song or not re.fullmatch(r'[A-Za-z0-9_]+',song):raise ValueError('当前入口需要有效歌曲代码 (manifest.song_selection.code)')
    unit=read_json(safe_file(task,f'data/{song}_unit.json'))['unitIdolData']
    anim_slots,lip_syncs,binding_source=_scenecreate_for_song(task,song)
    selected_positions=list(selection.get('slots') or range(1,len(unit)+1))
    motion_to_slots={}
    for pos in selected_positions:
        if not 1<=pos<=len(unit):raise ValueError('站位超出 unitIdolData 数组')
        motion_idx=anim_slots[pos-1] if anim_slots and pos-1<len(anim_slots) else pos
        motion_to_slots.setdefault(motion_idx,[]).append(pos)
    slots=[]
    for clip in report['clips']:
        if clip.get('kind') in ('body_with_root','auxiliary'):continue
        path=safe_file(task,clip['file']); source=sources.get(path.name.casefold())
        if source is None:source=sources.get(path.stem.casefold())
        if not source or not source.get('humanoid_bindings'):continue
        resource=source['resource']
        m=re.fullmatch(re.escape(song)+r'_(\d+)(?:_body\.asset|\.anim)',resource,re.I)
        # Some songs use a shared body clip plus numbered root tracks, or
        # permute slot -> motion/root/exp index in scenecreate (e.g. 110_mellow).
        if m:
            motion_idx=int(m[1])
            pairs=[(pos,motion_idx) for pos in motion_to_slots.get(motion_idx,[] if selection.get('slots') else [motion_idx])]
        elif re.fullmatch(re.escape(song)+r'(?:_body)?\.(?:asset|anim)',resource,re.I):
            pairs=[(pos,pos) for pos in selected_positions]
        else:continue
        for slot,motion_idx in pairs:
            if not 1<=slot<=len(unit):raise ValueError('站位超出 unitIdolData 数组')
            roots=[p for p in (task/'root-data').glob('*.unity3d')
                   if re.fullmatch(re.escape(song)+f'_{motion_idx:02d}_root',p.stem,re.I)]
            if not roots and len(unit)==1:
                roots=[p for p in (task/'root-data').glob('*.unity3d') if p.stem.casefold()==song.casefold()+'_root']
            if len(roots)!=1:raise ValueError(f'站位 {slot} 根位移文件不存在或不唯一')
            exps=[p for p in (task/'data').glob('*.json')
                  if re.fullmatch(re.escape(song)+f'_{motion_idx:02d}_exp',p.stem,re.I)]
            if not exps and len(unit)==1:
                exps=[p for p in (task/'data').glob('*.json') if p.stem.casefold()==song.casefold()+'_exp']
            if len(exps)!=1:raise ValueError(f'站位 {slot} 表情文件不存在或不唯一')
            exp=safe_file(task,f'data/{exps[0].name}')
            entry=dict(slot=slot,motion_slot=motion_idx,character_id=unit[slot-1]['chara_id'],body=str(path),
                       body_sha256=clip.get('sha256'),root=str(roots[0]),expression=str(exp),
                       duration=float(source['stop_time']),sample_rate=source['sample_rate'])
            if lip_syncs and slot-1<len(lip_syncs) and lip_syncs[slot-1]:
                entry['lip_sync']=lip_syncs[slot-1]
            slots.append(entry)
    if not slots or len({s['slot'] for s in slots})!=len(slots):
        raise ValueError('无法唯一确定各站位的原始 Humanoid 主动作')
    mouths=[str(safe_file(task,f)) for f in job.get('mouth_files',[])]
    camera=task/'data'/f'{song}_camera.json'
    warnings=[]
    sequence=task/'data'/f'{song}_sequence.json'
    cuts=read_json(sequence).get('cutSceneData',[]) if sequence.is_file() else []
    if cuts:warnings.append('本次转换歌曲主舞蹈；sequence 中的 UCS 演出切段不替换主动作，完整 MV 切段重建尚未支持。')
    return dict(schema=1,task_dir=str(task),task_id=task.name,song=song,slots=sorted(slots,key=lambda s:s['slot']),
                slot_binding_source=binding_source,
                mouths=mouths,camera=str(camera) if camera.is_file() else None,sequence_cuts=cuts,warnings=warnings,
                manifest_sha256=sha(task/'manifest.json'))


def root_curves(path):
    import UnityPy
    trees=[o.read_typetree() for o in UnityPy.load(str(path)).objects if o.type.name=='MonoBehaviour']
    roots=[t['rootCurveData'] for t in trees if 'rootCurveData' in t]
    if len(roots)!=1:raise ValueError('根位移数据缺失或不唯一')
    data=roots[0]
    if data.get('root')!=0:raise ValueError('尚未支持的 rootCurveData.root 模式')
    if data.get('rootScaling') not in (0,1,False,True):raise ValueError('无效 rootScaling 标记')
    for k in ['pos_y','rot_x','rot_y','rot_z','rot_w']:
        if data.get(k,{}).get('m_Curve'):raise ValueError('包含未经验证的独立根高度/旋转曲线: '+k)
    if not all(data.get('pos_'+a,{}).get('m_Curve') for a in 'xz'):
        raise ValueError('缺少 X/Z 根曲线')
    return data
