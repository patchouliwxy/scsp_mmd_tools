"""Streaming Unity Animator curves and Hermite/weighted Bezier sampling."""
from array import array
from pathlib import Path
import math
import numpy as np
from .timing import frame_count


def sample(curve, times):
    keys = curve if isinstance(curve,list) else curve.get('m_Curve', curve)
    if isinstance(keys, list):
        if not keys:
            raise ValueError('Empty animation curve')
        columns = {k: np.array([x.get(k, default) for x in keys], dtype=float)
                   for k, default in [('time',0), ('value',0), ('inSlope',0), ('outSlope',0),
                                      ('weightedMode',0), ('inWeight',1/3), ('outWeight',1/3)]}
    else:
        columns = {k: np.asarray(v, dtype=float) for k,v in keys.items()}
    t, v = columns['time'], columns['value']; times = np.asarray(times, dtype=float)
    if not len(t) or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Curve requires finite values and strictly increasing times')
    if len(t) == 1:
        return np.full(times.shape, v[0])
    a = np.clip(np.searchsorted(t,times,side='right')-1,0,len(t)-2); b = a+1
    dt = t[b]-t[a]; u = np.clip((times-t[a])/dt,0,1)
    si, so = columns['inSlope'], columns['outSlope']
    if np.isnan(si).any() or np.isnan(so).any():
        raise ValueError('NaN curve tangent')
    step = np.isinf(so[a]) | np.isinf(si[b])
    slope0 = np.where(step,0,so[a]); slope1 = np.where(step,0,si[b])
    result = ((2*u**3-3*u**2+1)*v[a] + (u**3-2*u**2+u)*slope0*dt
              + (-2*u**3+3*u**2)*v[b] + (u**3-u**2)*slope1*dt)
    mode = columns.get('weightedMode', np.zeros(len(t))).astype(int)
    weighted = ((mode[a]&2)!=0)|((mode[b]&1)!=0)
    if weighted.any():
        wo = np.where(mode[a]&2, columns['outWeight'][a], 1/3)
        wi = np.where(mode[b]&1, columns['inWeight'][b], 1/3)
        if np.any((wo<0)|(wo>1)|(wi<0)|(wi>1)):
            raise ValueError('Unsupported Bezier weight outside 0..1')
        low=np.zeros_like(u); high=np.ones_like(u)
        for _ in range(35):
            z=(low+high)/2
            bx=3*(1-z)**2*z*wo+3*(1-z)*z*z*(1-wi)+z**3
            low=np.where(bx<u,z,low); high=np.where(bx<u,high,z)
        z=(low+high)/2
        bv=(1-z)**3*v[a]+3*(1-z)**2*z*(v[a]+wo*dt*slope0)+3*(1-z)*z*z*(v[b]-wi*dt*slope1)+z**3*v[b]
        result=np.where(weighted,bv,result)
    result=np.where(step & (times<t[b]),v[a],result)
    return np.where(times<=t[0],v[0],np.where(times>=t[-1],v[-1],result))


def animator_curves(path):
    tracks={}; active=False; current=None; start=0.; stop=None; name=Path(path).stem
    defaults={'time':0,'value':0,'inSlope':0,'outSlope':0,'weightedMode':0,'inWeight':1/3,'outWeight':1/3}
    with Path(path).open(encoding='utf-8-sig') as src:
        for line in src:
            if line.startswith('  m_Name:'): name=line.split(':',1)[1].strip()
            if line.startswith('    m_StartTime:'): start=float(line.split(':',1)[1])
            if line.startswith('    m_StopTime:'): stop=float(line.split(':',1)[1])
            if line.startswith('  m_FloatCurves:'): active=True; continue
            if line.startswith('  m_PPtrCurves:'): active=False
            if not active: continue
            if line.startswith('  - serializedVersion:'):
                current={k:array('d') for k in defaults}
            elif current is not None:
                key,sep,value=line.strip().partition(':')
                if key in defaults:
                    if key=='time':
                        for k,default in defaults.items():
                            if k!='time' and len(current[k])<len(current['time']):current[k].append(default)
                    current[key].append(float(value.strip().lower().replace('.inf','inf').replace('.nan','nan')))
                elif key=='attribute': current['name']=value.strip().strip('"\'')
                elif key=='classID':
                    if int(value)==95 and current.get('name') and len(current['time']):
                        for k,default in defaults.items():
                            if len(current[k])<len(current['time']):current[k].append(default)
                            if len(current[k])!=len(current['time']):raise ValueError('Mismatched curve columns')
                        n=current.pop('name')
                        if n in tracks:raise ValueError('Duplicate Animator channel: '+n)
                        tracks[n]=current
                    current=None
    if stop is None or not math.isfinite(stop-start) or not 0<stop-start<=600:
        raise ValueError('Unsupported animation duration (expected 0..600 seconds)')
    return dict(name=name,start=start,duration=stop-start,tracks=tracks)


def normalize(name):
    return name.replace('LeftHand.','Left ').replace('RightHand.','Right ').replace('.',' ')


def humanoid_input(path, model, fps=30):
    clip=animator_curves(path); tracks=clip['tracks']; allowed={normalize(n) for n in model['muscles']}
    selected=[n for n in tracks if normalize(n) in allowed or n.startswith(('RootT.','RootQ.'))]
    if not any(normalize(n) in allowed for n in selected):raise ValueError('Not a Humanoid muscle clip')
    count=frame_count(clip['duration'],fps)
    if count>18001:raise ValueError('PoseWorker frame limit exceeded; use 30 fps or a shorter clip')
    times=np.minimum(np.arange(count)/fps,clip['duration'])+clip['start']
    data=np.stack([sample(tracks[n],times) for n in selected],axis=1)
    return dict(name=clip['name'],fps=fps,duration=clip['duration'],frameCount=count,channels=selected,
                samples=data.ravel().tolist(),omittedChannels=[n for n in tracks if n not in selected])
