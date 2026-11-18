"""Downloader song task -> standard VMD; reusable inspect_task/build_task API."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import traceback
import uuid
import numpy as np
from scsp_motion.backend import BuiltinPoseBackend,PoseWorkerBackend,read_json,write_json,sha
from scsp_motion.config import ROOT,load_rules
from scsp_motion.task import inspect_task,root_curves
from scsp_motion.curves import sample,humanoid_input
from scsp_motion.timing import plan_song_timeline,hold_pose_tail
from scsp_motion.retarget import load_target,load_generic_target,convert_pose
from scsp_motion.expressions import mouth_for,bake_morphs,bake_gaze
from scsp_motion.camera import bake_camera
from scsp_motion import vmd


def build_task(task_dir,output_root,*,backend_path=None,target_pmx=None,rules_path=None,
               slots=None,progress=None,backend=None):
    """target_pmx: None (generic), one PMX path, or {slot: PMX|None}. Returns schema-1 JSON report.

    A caller may provide a backend object with model, identity, calibrate(work)
    and solve(request,work) to replace the external Unity runtime entirely.
    progress follows scsp_mmd_tools: callback(percent=int, message=str).
    """
    info=inspect_task(task_dir);rules=load_rules(rules_path)
    backend=backend or (PoseWorkerBackend(backend_path) if (backend_path and str(backend_path).strip()) else BuiltinPoseBackend())
    selected=set(slots or [s['slot'] for s in info['slots']])
    if not selected<=set(s['slot'] for s in info['slots']):raise ValueError('选择了不存在的站位')
    generic_profile=ROOT/'Resources/generic_mmd_profile.json'
    if target_pmx is None or (isinstance(target_pmx,dict) and not target_pmx):
        targets={s:None for s in selected}
    elif isinstance(target_pmx,dict):
        targets={int(k):(Path(v).resolve(strict=True) if v else None) for k,v in target_pmx.items()}
    else:
        targets={s:Path(target_pmx).resolve(strict=True) for s in selected}
    if not selected<=targets.keys():raise ValueError('部分站位没有指定目标 PMX')
    models={s:(load_target(targets[s]) if targets[s] is not None else load_generic_target(generic_profile)) for s in selected}
    uses_generic=any(targets[s] is None for s in selected)
    outroot=Path(output_root).resolve()
    input_root=Path(info['task_dir'])
    if outroot.is_relative_to(input_root):raise ValueError('输出目录不能位于下载任务内')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=outroot/(info['song']+'_'+stamp);out.mkdir(parents=True,exist_ok=False)
    work=out/'_work';work.mkdir()
    gaze_note=('眼眉口事件包络为兼容近似；默认保留源眼骨动作，未验证的 ChangeTarget 转眼事件记为未还原（可显式设 gaze_mode=approximate），tongue/db 未映射，动态高光/Shader 不属于 VMD。'
               if rules['gaze_mode']=='source' else
               '眼眉口事件包络和视线为兼容近似，tongue/db 未映射，动态高光/Shader 不属于 VMD。')
    limitations=['主体重定向按目标 PMX 静息骨段校正；不保证异骨长模型足底接触完全一致。',
                 gaze_note,
                 '启用 IK 时脚踝/足趾目标由模型求解，膝盖方向可能因模型限角偏离 FK；默认 FK 保留完整骨骼姿态。']
    if uses_generic:
        limitations.append('部分或全部站位使用内置通用 MMD（45° A-pose）骨架模板导出；若目标模型身高差异较大或采用 T-pose/其他静息角/非标准骨链，请指定对应目标 PMX。')
    report=dict(schema=1,status='running',author='魔法少女labubu/patchouliwxy',input=info,output_dir=str(out),rules=rules,
                backend=backend.identity,characters=[],warnings=list(info['warnings']),
                validation=dict(binary_verified=False,native_mmd_playback_verified=False),
                limitations=limitations)
    def emit(p,m):
        if progress:progress(percent=int(p),message=m)
    def save():write_json(out/'转换报告.json',report)
    save()
    try:
        emit(2,'校准动作解算器')
        cal=backend.calibrate(work)
        basis=np.asarray(cal['root_xz_to_world'])*np.asarray(rules['source_axes'])
        # Evaluate origins for ALL source slots, even when exporting a subset.
        # This keeps group placement/cameras identical across separate exports.
        roots={}; starts=[]
        for slot in info['slots']:
            root=root_curves(slot['root']);roots[slot['slot']]=root
            starts.append([sample(root['pos_'+a],[0])[0] for a in 'xz'])
        origin=np.mean(starts,axis=0)@basis if rules['formation_origin']=='group' else np.zeros(3)
        report['formation']=dict(origin_world=origin.tolist(),policy=rules['formation_origin'],
                                 source_starts=starts,root_xz_to_mmd=basis.tolist())
        bone_names=read_json(ROOT/'Resources/mmd_bone_names.json')
        morph_names=read_json(ROOT/'Resources/unity_morph_names.json')
        char_file=next((ROOT/'Resources'/n for n in ('character_names.json','character-names.json') if (ROOT/'Resources'/n).is_file()),None)
        char_names=read_json(char_file).get('characters',{}) if char_file is not None else {}
        timeline=plan_song_timeline(info['slots'],roots,rules['fps'],rules['duration_policy'])
        report['timeline']=timeline
        count=timeline['frame_count'];times=np.arange(count)/rules['fps']
        for number,timing in timeline['slots'].items():
            if timing['body_hold_tail_seconds']>.001:
                report['warnings'].append(f"站位 {number}: 身体动作和根位移末帧保持约 {timing['body_hold_tail_seconds']:.3f} 秒以对齐歌曲时间轴；未重建缺失的后续舞蹈或 UCS 切段。")
        save()
        solved={}
        for index,slot in enumerate(s for s in info['slots'] if s['slot'] in selected):
            s=slot['slot'];emit(8+index*55/len(selected),f'站位 {s}: 解析 Humanoid 曲线与解算骨骼')
            if slot.get('body_sha256') and sha(slot['body'])!=slot['body_sha256']:raise ValueError('身体动作哈希与导出报告不符')
            request=humanoid_input(Path(slot['body']),backend.model)
            root=roots[s]
            for a in 'xz':
                end=root['pos_'+a]['m_Curve'][-1]['time']
                if abs(end-request['duration'])>.05:raise ValueError('身体动作和根位移时长不一致')
            if any(c in request['channels'] for c in ('RootT.x','RootT.z')):
                raise ValueError('原始身体动作已有水平根轨迹，拒绝再叠加 root-data')
            p,q,meta=backend.solve(request,work/f'slot_{s:02d}')
            p,q=hold_pose_tail(p,q,count)
            timing=dict(timeline['slots'][str(s)],body_duration=request['duration'],
                        source_frame_count=request['frameCount'],output_frame_count=count,
                        body_hold_tail_seconds=max(0.,timeline['duration']-request['duration']))
            if count>request['frameCount']:
                emit(8+index*55/len(selected),f"站位 {s}: 保持原时间采样，末帧补齐到共同时间轴（约 {timing['body_hold_tail_seconds']:.3f} 秒）")
            root_p=np.stack([sample(root['pos_'+a],times) for a in 'xz'],axis=1)
            displacement=(root_p@basis-origin)*rules['units_per_meter']
            tracks,ik,retarget,world=convert_pose(models[s],backend.model,cal,p,q,displacement,rules,bone_names)
            checkpoints=sorted(set([0,min(30,count-1),min(150,count-1),min(900,count-1),count-1]))
            verify_names=set(retarget['mapped_bones'])-({'左目','右目'} if rules['gaze_mode']=='approximate' else set())
            expected={str(f):{b.name:world[i][f].tolist() for i,b in enumerate(models[s].bones) if b.name in verify_names} for f in checkpoints}
            write_json(work/f'slot_{s:02d}'/'expected-fk-samples.json',expected)
            solved[s]=dict(slot=slot,bones=tracks,ik=ik,retarget=retarget,world=world,
                           duration=request['duration'],timing=timing,source_omitted_channels=request['omittedChannels'])
        cameras=None;camera_pos=None
        if rules['camera_mode']!='off' and info['camera']:
            emit(68,'转换相机采样与镜头切换')
            actor_anchors={s:{b.name:d['world'][i] for i,b in enumerate(models[s].bones)
                              if b.name in ('頭','首','上半身3','上半身2','上半身','下半身','センター','左手首','左ひじ','右手首','右ひじ')}
                           for s,d in solved.items()}
            slot_characters={s['slot']:s['character_id'] for s in info['slots']}
            cameras,camera_pos,camreport=bake_camera(info['camera'],count,origin,basis,rules,
                                                     actor_positions=actor_anchors,slot_characters=slot_characters)
            checkpoints=sorted(set([0,min(30,count-1),min(150,count-1),min(900,count-1),count-1]))
            write_json(work/'expected-camera-samples.json',{str(f):dict(position=camera_pos[f].tolist()) for f in checkpoints})
            path=out/(info['song']+'_camera.vmd')
            validation=vmd.write(path,'カメラ・照明',cameras=cameras)
            report['camera']=dict(file=str(path),sha256=sha(path),validation=validation,**camreport)
        elif rules['camera_mode']!='off':report['warnings'].append('输入未提供相机数据，未生成相机 VMD。')
        all_heads={s:d['world'][next(i for i,b in enumerate(models[s].bones) if b.name=='頭')] for s,d in solved.items()}
        for index,(s,d) in enumerate(solved.items()):
            emit(72+index*25/len(selected),f'站位 {s}: 烘焙表情、口型与眼部并写入 VMD')
            mouth=mouth_for(d['slot'],info,rules)
            morphs,exp_report,gaze=bake_morphs(d['slot']['expression'],mouth,count,rules,models[s],morph_names)
            eye_tracks,gaze_report=bake_gaze(gaze,models[s],d['world'],d['bones'],all_heads,camera_pos,rules)
            d['bones'].update(eye_tracks)
            path=out/f"{info['song']}_slot{s:02d}_chara{d['slot']['character_id']:02d}.vmd"
            name=''
            for character in models[s].name:
                try:encoded=character.encode('cp932')
                except UnicodeEncodeError:character='_';encoded=b'_'
                if len(name.encode('cp932'))+len(encoded)>20:break
                name+=character
            name=name.strip() or 'SCSP model'
            validation=vmd.write(path,name,d['bones'],morphs,ik=d['ik'])
            target_ref=str(targets[s]) if targets[s] is not None else 'builtin:generic-mmd'
            target_hash=sha(targets[s]) if targets[s] is not None else sha(generic_profile)
            cid=d['slot']['character_id']
            cname=char_names.get(str(cid)) or char_names.get(f'{cid:02d}','')
            report['characters'].append(dict(slot=s,motion_slot=d['slot'].get('motion_slot',s),
                     character_id=cid,character_name=cname,file=str(path),sha256=sha(path),
                     target_mode=('pmx' if targets[s] is not None else 'generic'),
                     target_pmx=target_ref,target_pmx_sha256=target_hash,mouth=str(mouth),validation=validation,
                     retarget=d['retarget'],expressions=exp_report,gaze=gaze_report,duration=d['duration'],
                     output_duration=timeline['duration'],timing=d['timing'],
                     source_omitted_channels=d['source_omitted_channels'],
                     source_hashes={k:sha(d['slot'][k]) for k in ('body','root','expression')}))
            save()
        report['status']='completed';report['validation']['binary_verified']=True
        ik_note=('本次为 FK 模式，已关闭足/脚趾 IK；不要手动开启 IK 叠加解算。' if rules['ik_mode']=='fk' else
                 '本次为 IK 模式，足/脚趾 IK 已启用；膝盖由目标模型限角重新求解，可能偏离 FK。')
        generic_note='包含使用内置通用 MMD（45° A-pose）骨架模板导出的动作；适合标准 MMD 模型通用套用，异体型/其他静息姿态请指定对应 PMX。\n' if uses_generic else ''
        timing_note='各站位保持原采样速度；身体或根位移较短时保持各自末帧至共同时间轴结束，表情口型按原时间继续。补齐区间详见报告 timeline/timing；未重建缺失舞蹈或 UCS。\n'
        roster_lines='\n'.join(
            f"- {Path(c['file']).name}: 站位 {c['slot']} · 角色 {c['character_id']:02d}"
            +(f" ({c['character_name']})" if c.get('character_name') else '')
            +(f" [动作源编号 {c['motion_slot']:02d}]" if c.get('motion_slot')!=c['slot'] else '')
            for c in report['characters']
        )
        roster_note=('本次导出角色文件对照：\n'+roster_lines+'\n\n') if roster_lines else ''
        (out/'使用说明.txt').write_text('exported by scsp_mmd_tools\nauthor:魔法少女labubu/patchouliwxy\n\n'+roster_note+'每站位的人物 VMD 载入对应目标 PMX（通用导出适用于标准 45° A-pose MMD 模型）；camera.vmd 载入相机/照明。\n所有 VMD 固定 30 FPS；多人使用共同原点，分开导出也保留同一站位。\n'+timing_note+ik_note+'\n'+generic_note+'请使用同任务的 audio WAV 对齐时间，不从表情尾哨兵推算歌长。\n已做格式校验；实际动作、接触、表情和镜头观感由用户验收。\n已知近似与缺项见转换报告.json。\n',encoding='utf-8-sig')
        save();emit(100,'转换完成: '+str(out));return report
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc());save()
        raise RuntimeError(f'转换失败: {error}\n报告: {out / "转换报告.json"}') from error


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('task')
    p=sub.add_parser('build');p.add_argument('task');p.add_argument('--output',required=True)
    p.add_argument('--backend',required=False,default='',help='下载器目录或 PoseWorker 目录；省略时使用内置解算器')
    p.add_argument('--pmx',action='append',required=False,help='目标 PMX，或 站位=PMX；可多次指定，省略时按内置通用 MMD 骨架导出')
    p.add_argument('--rules');p.add_argument('--slot',type=int,action='append');p.add_argument('--report')
    p=sub.add_parser('inspect-vmd');p.add_argument('path')
    args=parser.parse_args(argv)
    if args.command=='inspect':result=inspect_task(args.task)
    elif args.command=='inspect-vmd':result=vmd.inspect(args.path)
    else:
        if not args.pmx:pmx=None
        elif len(args.pmx)==1 and '=' not in args.pmx[0]:pmx=args.pmx[0]
        else:pmx={int(s.split('=',1)[0]):(s.split('=',1)[1] or None) for s in args.pmx}
        result=build_task(args.task,args.output,backend_path=args.backend,target_pmx=pmx,rules_path=args.rules,slots=args.slot,
                          progress=lambda **v:print(json.dumps(v,ensure_ascii=False),flush=True))
        if args.report:write_json(args.report,result)
        result={k:result[k] for k in ('status','output_dir','validation')}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'):sys.stderr.reconfigure(encoding='utf-8')
    try:raise SystemExit(main())
    except Exception as error:print(str(error),file=sys.stderr);raise SystemExit(1)
