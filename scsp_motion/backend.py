"""Pose solver adapters (built-in pure-Python Humanoid solver and external PoseWorker)."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np
from .math3d import inv, mul, rotate, unit, world_pose


def read_json(path):
    def bad(value):raise ValueError('Non-finite JSON number: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=bad)


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as src:
        for b in iter(lambda:src.read(1048576),b''):h.update(b)
    return h.hexdigest()


class PoseWorkerBackend:
    def __init__(self,path):
        p=Path(path).expanduser().resolve()
        if p.is_file():p=p.parent
        if (p/'tools/PoseWorker').is_dir():p=p/'tools/PoseWorker'
        self.directory=p; self.exe=p/'SCSPPoseWorker.exe'; self.model_path=p/'model.json'
        if not self.exe.is_file() or not self.model_path.is_file():
            raise ValueError('请选择含 tools/PoseWorker 的下载器目录，或 PoseWorker 目录')
        self.model=read_json(self.model_path)
        if not self.model.get('humanoid') or not self.model.get('nodes') or not self.model.get('muscles'):
            raise ValueError('PoseWorker model.json 缺少有效 Humanoid 骨架')
        self.identity=dict(type='external_poseworker',exe=str(self.exe),exe_sha256=sha(self.exe),
                           model_sha256=sha(self.model_path),model_id=self.model.get('id'))

    def solve(self,request,work):
        work=Path(work).resolve(); work.mkdir(parents=True,exist_ok=True)
        if work.is_relative_to(self.directory) or self.directory.is_relative_to(work):
            raise ValueError('Worker 工作目录必须独立于下载器运行时')
        source=work/'request.json'; target=work/'pose'
        # Compact serialization keeps full-song input bounded.
        source.write_text(json.dumps(request,separators=(',',':'),allow_nan=False),encoding='utf-8')
        args=[str(self.exe),'-batchmode','-nographics','--pose-input',str(source),
              '--pose-out',str(target),'-logFile',str(work/'unity.log')]
        with (work/'process.log').open('wb') as log:
            result=subprocess.run(args,cwd=work,stdout=log,stderr=subprocess.STDOUT,
                                  creationflags=0x08000000 if os.name=='nt' else 0,timeout=300)
        if result.returncode or not target.with_suffix('.json').is_file():
            raise RuntimeError('PoseWorker 解算失败，请查看 '+str(work/'unity.log'))
        meta=read_json(target.with_suffix('.json'))
        if meta['frameCount']!=request['frameCount'] or not 1<=meta['boneCount']<=len(self.model['nodes']):
            raise ValueError('Worker 返回帧数/骨骼数异常')
        indices=meta['nodes']
        if len(indices)!=meta['boneCount'] or len(set(indices))!=len(indices) or any(not 0<=i<len(self.model['nodes']) for i in indices):
            raise ValueError('Worker 返回节点索引异常')
        data=np.fromfile(target.with_suffix('.bin'),dtype='<f4')
        expected=meta['frameCount']*meta['boneCount']*7
        if len(data)!=expected or not np.isfinite(data).all():raise ValueError('Worker 姿态文件无效')
        data=data.reshape(meta['frameCount'],meta['boneCount'],7)
        write_json(work/'backend.json',dict(**self.identity,request_sha256=sha(source),
                                         metadata=meta,source_omitted_channels=request.get('omittedChannels',[])))
        p,q=world_pose(self.model['nodes'],data,indices)
        return p,q,meta

    def calibrate(self,work):
        # Zero muscles establish an explicit neutral pose. Three unit body
        # translations measure Avatar body-to-world axes and humanScale.
        channels=list(self.model['muscles'])+['RootT.x','RootT.z']
        values=np.zeros((3,len(channels))); values[1,-2]=1; values[2,-1]=1
        req=dict(name='converter-calibration',fps=30,duration=2/30,frameCount=3,
                 channels=channels,samples=values.ravel().tolist())
        p,q,meta=self.solve(req,Path(work)/'calibration')
        hips=next(i for i,n in enumerate(self.model['nodes']) if n['name']=='Hips')
        basis=np.stack((p[1,hips]-p[0,hips],p[2,hips]-p[0,hips]))
        scale=float(self.model['humanScale'])
        if not .01<scale<10 or np.max(np.abs(np.linalg.norm(basis,axis=1)-scale))>1e-3:
            raise ValueError('Avatar body translation calibration disagrees with humanScale')
        if abs(np.dot(*basis))>1e-3:raise ValueError('Avatar body axes are not orthogonal')
        # Zero muscles are mid-range joint values, NOT the source T bind pose.
        # Retarget rest correction must use the actual exported bind hierarchy.
        rest_p,rest_q=world_pose(self.model['nodes'])
        result=dict(neutral_positions=rest_p[0].tolist(),neutral_rotations=rest_q[0].tolist(),
                    zero_muscle_positions=p[0].tolist(),
                    root_xz_to_world=(basis/scale).tolist(),human_scale=scale)
        write_json(Path(work)/'calibration.json',result)
        return result


def _axis_angle_batch(vecs):
    ang=np.linalg.norm(vecs,axis=-1,keepdims=True)
    small=ang<1e-12
    half=0.5*ang
    scale_v=np.where(small,0.5,np.sin(half)/np.where(small,1.,ang))
    return np.concatenate((vecs*scale_v,np.cos(half)),axis=-1)


def _basis_to_quat(R):
    """Convert (N, 3, 3) orthonormal right-handed basis matrices to xyzw quaternions."""
    t=R[:,0,0]+R[:,1,1]+R[:,2,2]
    q=np.zeros((len(R),4))
    m0=t>0
    if m0.any():
        s0=np.sqrt(np.maximum(t[m0]+1.,1e-12))*2.
        q[m0,3]=0.25*s0
        q[m0,0]=(R[m0,2,1]-R[m0,1,2])/s0
        q[m0,1]=(R[m0,0,2]-R[m0,2,0])/s0
        q[m0,2]=(R[m0,1,0]-R[m0,0,1])/s0
    m1=(~m0)&(R[:,0,0]>R[:,1,1])&(R[:,0,0]>R[:,2,2])
    if m1.any():
        s1=np.sqrt(np.maximum(1.+R[m1,0,0]-R[m1,1,1]-R[m1,2,2],1e-12))*2.
        q[m1,3]=(R[m1,2,1]-R[m1,1,2])/s1
        q[m1,0]=0.25*s1
        q[m1,1]=(R[m1,0,1]+R[m1,1,0])/s1
        q[m1,2]=(R[m1,0,2]+R[m1,2,0])/s1
    m2=(~m0)&(~m1)&(R[:,1,1]>R[:,2,2])
    if m2.any():
        s2=np.sqrt(np.maximum(1.+R[m2,1,1]-R[m2,0,0]-R[m2,2,2],1e-12))*2.
        q[m2,3]=(R[m2,0,2]-R[m2,2,0])/s2
        q[m2,0]=(R[m2,0,1]+R[m2,1,0])/s2
        q[m2,1]=0.25*s2
        q[m2,2]=(R[m2,1,2]+R[m2,2,1])/s2
    m3=(~m0)&(~m1)&(~m2)
    if m3.any():
        s3=np.sqrt(np.maximum(1.+R[m3,2,2]-R[m3,0,0]-R[m3,1,1],1e-12))*2.
        q[m3,3]=(R[m3,1,0]-R[m3,0,1])/s3
        q[m3,0]=(R[m3,0,2]+R[m3,2,0])/s3
        q[m3,1]=(R[m3,1,2]+R[m3,2,1])/s3
        q[m3,2]=0.25*s3
    return unit(q)


class BuiltinPoseBackend:
    """Pure-Python/NumPy Unity Humanoid Muscle -> world pose solver using generic kinematics."""

    def __init__(self,spec_path=None):
        from .config import ROOT
        from .curves import normalize
        path=Path(spec_path) if spec_path else (ROOT/'Resources/humanoid_avatar.json')
        spec=read_json(path)
        self.spec=spec
        self.model=dict(schema=spec['schema'],id=spec['id'],humanoid=spec['humanoid'],
                        humanScale=spec['humanScale'],nodes=spec['nodes'],muscles=spec['muscles'])
        self.identity=dict(type='builtin_humanoid_solver',spec_sha256=sha(path),id=spec['id'])
        self._out=spec['output_nodes']
        self._bnames=[spec['nodes'][i]['name'] for i in self._out]
        self._out_map={idx:bi for bi,idx in enumerate(self._out)}
        self._hips_bi=self._bnames.index('Hips')
        self._hips_idx=self._out[self._hips_bi]
        self._larm=self._out[self._bnames.index('LeftArm')]
        self._rarm=self._out[self._bnames.index('RightArm')]
        self._lleg=self._out[self._bnames.index('LeftUpLeg')]
        self._rleg=self._out[self._bnames.index('RightUpLeg')]
        self._com_w=np.asarray(spec['com_weights'],dtype=float)
        self._zpos=np.asarray(spec['zero_local_positions'],dtype=float)
        self._zrot=np.asarray(spec['zero_local_rotations'],dtype=float)
        self._base_m=np.asarray(spec['baseline_muscles'],dtype=float)
        self._neut_p=np.asarray(spec['neutral_body_position'],dtype=float)
        self._neut_q=np.asarray(spec['neutral_body_rotation'],dtype=float)
        self._norm_to_mi={normalize(m):i for i,m in enumerate(spec['muscles'])}
        self._normalize=normalize

    def solve(self,request,work=None):
        n=int(request['frameCount'])
        ch=list(request['channels'])
        if n<1 or not ch:
            raise ValueError('无效动作请求帧数或通道')
        samples=np.asarray(request['samples'],dtype=float)
        if samples.size!=n*len(ch) or not np.isfinite(samples).all():
            raise ValueError('动作采样维度或数值无效')
        samples=samples.reshape(n,len(ch))
        m_mat=np.tile(self._base_m,(n,1))
        root_t=np.tile(self._neut_p,(n,1))
        root_q=np.tile(self._neut_q,(n,1))
        for ci,name in enumerate(ch):
            nc=self._normalize(name)
            if nc in self._norm_to_mi:
                m_mat[:,self._norm_to_mi[nc]]=samples[:,ci]
            elif name=='RootT.x':root_t[:,0]=samples[:,ci]
            elif name=='RootT.y':root_t[:,1]=samples[:,ci]
            elif name=='RootT.z':root_t[:,2]=samples[:,ci]
            elif name=='RootQ.x':root_q[:,0]=samples[:,ci]
            elif name=='RootQ.y':root_q[:,1]=samples[:,ci]
            elif name=='RootQ.z':root_q[:,2]=samples[:,ci]
            elif name=='RootQ.w':root_q[:,3]=samples[:,ci]
        root_q=unit(root_q)

        data=np.empty((n,len(self._out),7),dtype=float)
        data[:,:,:3]=self._zpos[None,:,:]
        for bi,terms in enumerate(self.spec['bone_muscle_terms']):
            if bi==self._hips_bi or not terms:
                data[:,bi,3:]=self._zrot[bi]
                continue
            lt=np.zeros((n,3));sw=np.zeros((n,3));rt=np.zeros((n,3))
            for t in terms:
                v=m_mat[:,t['muscle']]
                ang=np.deg2rad(v*np.where(v>=0,t['pos_deg'],t['neg_deg']))[:,None]
                ax=np.asarray(t['axis'],dtype=float)[None,:]
                role=t['role']
                if role=='left_twist':lt+=ang*ax
                elif role=='right_twist':rt+=ang*ax
                else:sw+=np.tan(0.5*ang)*ax
            q_sw=unit(np.concatenate((sw,np.ones((n,1))),axis=-1))
            qd=mul(q_sw,_axis_angle_batch(rt))
            if np.any(lt):qd=mul(_axis_angle_batch(lt),qd)
            data[:,bi,3:]=mul(self._zrot[bi],qd)

        nodes=self.model['nodes']
        n_nodes=len(nodes)
        cum_s=np.empty((n_nodes,3),dtype=float)
        in_hips={self._hips_idx}
        for i,node in enumerate(nodes):
            par=node['parent']
            ls=np.asarray(node['scale'],dtype=float)
            cum_s[i]=ls if par<0 else (cum_s[par]*ls)
            if par in in_hips:in_hips.add(i)

        hp=np.zeros((n,n_nodes,3))
        hq=np.zeros((n,n_nodes,4));hq[:,:,3]=1.
        for i,node in enumerate(nodes):
            if i==self._hips_idx or i not in in_hips:continue
            par=node['parent']
            bi=self._out_map.get(i)
            lp=data[:,bi,:3] if bi is not None else np.asarray(node['position'])
            lq=data[:,bi,3:] if bi is not None else np.asarray(node['rotation'])
            hp[:,i]=hp[:,par]+rotate(hq[:,par],cum_s[par]*lp)
            hq[:,i]=unit(mul(hq[:,par],lq))

        Y=unit((hp[:,self._rarm]+hp[:,self._larm])-(hp[:,self._rleg]+hp[:,self._lleg]))
        X_raw=(hp[:,self._rleg]-hp[:,self._lleg])+(hp[:,self._rarm]-hp[:,self._larm])
        Z=unit(np.cross(X_raw,Y))
        X=np.cross(Y,Z)
        q_body_in_hips=_basis_to_quat(np.stack((X,Y,Z),axis=-1))

        world_q_hips=mul(root_q,inv(q_body_in_hips))
        world_p_rel=rotate(world_q_hips[:,None,:],hp[:,self._out])
        com_rel=np.einsum('fbi,b->fi',world_p_rel,self._com_w)
        world_p_hips=root_t*float(self.model['humanScale'])-com_rel

        p_all=np.zeros((n,n_nodes,3))
        q_all=np.zeros((n,n_nodes,4));q_all[:,:,3]=1.
        rest_p,rest_q=world_pose(nodes)
        p_all[:]=rest_p;q_all[:]=rest_q
        sub=sorted(in_hips)
        p_all[:,sub]=world_p_hips[:,None,:]+rotate(world_q_hips[:,None,:],hp[:,sub])
        q_all[:,sub]=mul(world_q_hips[:,None,:],hq[:,sub])
        meta=dict(schema=1,name=request['name'],fps=request['fps'],duration=request['duration'],
                  frameCount=n,boneCount=len(self._out),nodes=self._out)
        if work is not None:
            work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
            write_json(work/'backend.json',dict(**self.identity,metadata=meta,
                                               source_omitted_channels=request.get('omittedChannels',[])))
        return p_all,q_all,meta

    def calibrate(self,work=None):
        channels=list(self.model['muscles'])+['RootT.x','RootT.z']
        values=np.zeros((3,len(channels)));values[1,-2]=1;values[2,-1]=1
        req=dict(name='converter-calibration',fps=30,duration=2/30,frameCount=3,
                 channels=channels,samples=values.ravel().tolist())
        p,q,_=self.solve(req,Path(work)/'calibration' if work is not None else None)
        basis=np.stack((p[1,self._hips_idx]-p[0,self._hips_idx],p[2,self._hips_idx]-p[0,self._hips_idx]))
        scale=float(self.model['humanScale'])
        if not .01<scale<10 or np.max(np.abs(np.linalg.norm(basis,axis=1)-scale))>1e-3:
            raise ValueError('Avatar body translation calibration disagrees with humanScale')
        if abs(np.dot(*basis))>1e-3:raise ValueError('Avatar body axes are not orthogonal')
        rest_p,rest_q=world_pose(self.model['nodes'])
        result=dict(neutral_positions=rest_p[0].tolist(),neutral_rotations=rest_q[0].tolist(),
                    zero_muscle_positions=p[0].tolist(),
                    root_xz_to_world=(basis/scale).tolist(),human_scale=scale)
        if work is not None:
            write_json(Path(work)/'calibration.json',result)
        return result
