"""Shape-preserving camera sampling and fractional-FOV distance compensation.

VMD shares all camera channels at each key, and its FOV field is an integer.
Compensation preserves the projection on a reference plane, not all scene depths.
"""
import functools
import numpy as np
from .math3d import euler_unity,rotate

LINEAR=(20,20,107,107)


def slopes(values):
    """PCHIP derivatives on a uniform grid, in units per sample."""
    values=np.asarray(values,dtype=float);out=np.zeros_like(values)
    if len(values)<2:return out
    delta=np.diff(values,axis=0)
    if len(values)==2:out[:]=delta[0];return out
    same=delta[:-1]*delta[1:]>0
    np.divide(2*delta[:-1]*delta[1:],delta[:-1]+delta[1:],out=out[1:-1],where=same)
    for idx,d0,d1 in [(0,delta[0],delta[1]),(-1,delta[-1],delta[-2])]:
        edge=(3*d0-d1)/2
        edge=np.where(edge*d0<=0,0,edge)
        out[idx]=np.where((d0*d1<0)&(abs(edge)>3*abs(d0)),3*d0,edge)
    return out


def _nonuniform_slopes(x,values):
    """Fritsch-Carlson PCHIP derivatives on strictly increasing 1D grid x."""
    h=np.diff(x);hb=h[:,None] if values.ndim>1 else h
    m=np.diff(values,axis=0)/hb;out=np.zeros_like(values)
    if len(values)==2:out[:]=m[0];return out
    same=m[:-1]*m[1:]>0
    w1=2*h[1:]+h[:-1];w2=h[1:]+2*h[:-1]
    if values.ndim>1:w1=w1[:,None];w2=w2[:,None]
    np.divide(w1+w2,w1/np.where(same,m[:-1],1.)+w2/np.where(same,m[1:],1.),out=out[1:-1],where=same)
    for idx,h0,h1,m0,m1 in [(0,h[0],h[1],m[0],m[1]),(-1,h[-1],h[-2],m[-1],m[-2])]:
        edge=((2*h0+h1)*m0-h0*m1)/(h0+h1)
        edge=np.where(edge*m0<=0,0,edge)
        out[idx]=np.where((m0*m1<=0)&(abs(edge)>3*abs(m0)),3*m0,edge)
    return out


def sample_series(values,times,*,grid_x=None):
    values=np.asarray(values,dtype=float);times=np.asarray(times,dtype=float)
    if len(values)==1:return np.repeat(values[:1],len(times),axis=0)
    if grid_x is not None:
        gx=np.asarray(grid_x,dtype=float)
        d=_nonuniform_slopes(gx,values)
        xc=np.clip(times,gx[0],gx[-1])
        i=np.clip(np.searchsorted(gx,xc,side='right')-1,0,len(values)-2)
        h=np.diff(gx)[i];u=(xc-gx[i])/h
        if values.ndim>1:h=h[:,None];u=u[:,None]
        return ((2*u**3-3*u**2+1)*values[i]+(u**3-2*u**2+u)*h*d[i]
                +(-2*u**3+3*u**2)*values[i+1]+(u**3-u**2)*h*d[i+1])
    s=slopes(values);x=np.clip(times,0,1)*(len(values)-1)
    i=np.minimum(x.astype(int),len(values)-2);u=x-i
    if values.ndim>1:u=u[:,None]
    return ((2*u**3-3*u**2+1)*values[i]+(u**3-2*u**2+u)*s[i]
            +(-2*u**3+3*u**2)*values[i+1]+(u**3-u**2)*s[i+1])


def channel_curve(delta,start_slope,end_slope,mid_delta=None):
    if abs(delta)<1e-12:return LINEAR
    y1=np.clip(start_slope/(3*delta),0,1)
    y2=np.clip(1-end_slope/(3*delta),0,1)
    if mid_delta is not None:
        target=np.clip((mid_delta/delta-.125)/.375,0,2)
        y1=np.clip(y1+(target-y1-y2)/2,0,1)
        y2=np.clip(target-y1,0,1);y1=np.clip(target-y2,0,1)
    return (42,int(round(y1*127)),85,int(round(y2*127)))


def pack(curves):
    return bytes(v for x1,y1,x2,y2 in curves for v in (x1,x2,y1,y2))


def _cap_shift(shift,max_shift):
    if max_shift is None:return shift
    if max_shift<=0:return np.zeros_like(shift)
    return max_shift*np.tanh(shift/max_shift)


def _eval_vmd_bezier(t,cands):
    """Evaluate cubic Bezier (0,0)->(x1,y1)->(x2,y2)->(1,1) for cands (K,4) at t (L,)."""
    x1=cands[:,0:1]/127.0;y1=cands[:,1:2]/127.0;x2=cands[:,2:3]/127.0;y2=cands[:,3:4]/127.0
    t=t[None,:]
    lo=np.zeros((len(cands),t.shape[1]));hi=np.ones_like(lo)
    for _ in range(14):
        u=0.5*(lo+hi);om=1.0-u
        x=3*om*om*u*x1+3*om*u*u*x2+u*u*u
        lo=np.where(x<t,u,lo);hi=np.where(x>=t,u,hi)
    u=0.5*(lo+hi);om=1.0-u
    return 3*om*om*u*y1+3*om*u*u*y2+u*u*u


@functools.lru_cache(maxsize=256)
def _monotonic_bezier_basis(length):
    """Precompute Bernstein basis at fixed monotonic X=(42/127, 85/127)."""
    ts=np.linspace(0.0,1.0,length)
    x1,x2=42.0/127.0,85.0/127.0
    lo=np.zeros(length);hi=np.ones(length)
    for _ in range(18):
        u=0.5*(lo+hi);om=1.0-u
        x=3*om*om*u*x1+3*om*u*u*x2+u*u*u
        lo=np.where(x<ts,u,lo);hi=np.where(x>=ts,u,hi)
    u=0.5*(lo+hi);om=1.0-u
    return ts,3*om*om*u,3*om*u*u,u*u*u


def _fit_bezier_multi(segs,endpoints=None):
    """Fit one monotonic VMD (x1,y1,x2,y2) curve in 0..127 (x1<=x2, y1<=y2)."""
    length=len(segs[0])
    v0s=[(s[0] if endpoints is None else endpoints[idx][0]) for idx,s in enumerate(segs)]
    v1s=[(s[-1] if endpoints is None else endpoints[idx][1]) for idx,s in enumerate(segs)]
    ts=np.linspace(0.0,1.0,length)
    err_lin=max(float(np.max(np.abs(v0+(v1-v0)*ts-s))) for v0,v1,s in zip(v0s,v1s,segs))
    if err_lin<1e-6 or length<=2:
        return LINEAR,err_lin
    _,b1,b2,b3=_monotonic_bezier_basis(length)
    a_rows=[];rhs_rows=[]
    for v0,v1,s in zip(v0s,v1s,segs):
        dv=v1-v0
        if abs(dv)<1e-6:continue
        a_rows.append(np.column_stack([dv*b1[1:-1],dv*b2[1:-1]]))
        rhs_rows.append(s[1:-1]-v0-dv*b3[1:-1])
    if not a_rows:
        return LINEAR,err_lin
    a_mat=np.vstack(a_rows);rhs=np.concatenate(rhs_rows)
    y,*_=np.linalg.lstsq(a_mat,rhs,rcond=None)
    # Collect unconstrained-clipped and active-set monotonic candidates (0 <= y1 <= y2 <= 1).
    y_pairs=[(float(np.clip(y[0],0.0,1.0)),float(np.clip(y[1],np.clip(y[0],0.0,1.0),1.0)))]
    if y[0]>y[1]:
        col=a_mat[:,0]+a_mat[:,1]
        denom=float(np.dot(col,col))
        if denom>1e-18:
            w=float(np.clip(np.dot(col,rhs)/denom,0.0,1.0))
            y_pairs.append((w,w))
    cands={LINEAR}
    for y1f,y2f in y_pairs:
        iy1=int(np.clip(round(127.0*y1f),0,127))
        iy2=int(np.clip(round(127.0*y2f),iy1,127))
        cands.add((42,iy1,85,iy2))
    best_curve=LINEAR;best_err=err_lin
    for curve in sorted(cands):
        if curve==LINEAR:
            continue
        bz=b1*(curve[1]/127.0)+b2*(curve[3]/127.0)+b3
        err=0.0
        for v0,v1,s in zip(v0s,v1s,segs):
            err=max(err,float(np.max(np.abs(v0+(v1-v0)*bz-s))))
        if err<best_err-1e-9:
            best_err=err;best_curve=curve
    return best_curve,float(best_err)


def reduce_block_keys(pos_vmd,rot_vmd_rad,fov_vmd,a,b,in_interp,*,
                      pos_samples=None,rot_samples_deg=None,fov_samples=None,factor=1,
                      pos_tol=0.008,rot_tol=0.02,fov_tol=0.51):
    """Fit sparse cubic Bezier VMD keys across integer frame range [a, b]."""
    rot_deg_vmd=np.rad2deg(rot_vmd_rad)
    p_s=pos_vmd if pos_samples is None else pos_samples
    r_s=rot_deg_vmd if rot_samples_deg is None else rot_samples_deg
    f_s=fov_vmd if fov_samples is None else fov_samples
    keys=[dict(frame=a,distance=0.,position=pos_vmd[a].tolist(),rotation=rot_vmd_rad[a].tolist(),
               fov=int(round(fov_vmd[a])),perspective=True,interpolation=in_interp)]
    i=a
    while i<b:
        best_j=i+1;best_curves=(LINEAR,)*5
        lo,hi=i+1,b
        while lo<=hi:
            mid=(lo+hi)//2
            if mid==i+1:
                best_j=mid;best_curves=(LINEAR,)*5;lo=mid+1
                continue
            ia,ib=i*factor,mid*factor+1
            bx,epx=_fit_bezier_multi([p_s[ia:ib,0]],endpoints=[(float(pos_vmd[i,0]),float(pos_vmd[mid,0]))])
            ok=epx<=pos_tol
            if ok:
                by,epy=_fit_bezier_multi([p_s[ia:ib,1]],endpoints=[(float(pos_vmd[i,1]),float(pos_vmd[mid,1]))])
                ok=epy<=pos_tol
            if ok:
                bz,epz=_fit_bezier_multi([p_s[ia:ib,2]],endpoints=[(float(pos_vmd[i,2]),float(pos_vmd[mid,2]))])
                ok=epz<=pos_tol
            if ok:
                br,er=_fit_bezier_multi([r_s[ia:ib,0],r_s[ia:ib,1],r_s[ia:ib,2]],
                                         endpoints=[(float(rot_deg_vmd[i,k]),float(rot_deg_vmd[mid,k])) for k in range(3)])
                ok=er<=rot_tol
            if ok:
                bf,ef=_fit_bezier_multi([f_s[ia:ib]],endpoints=[(float(round(fov_vmd[i])),float(round(fov_vmd[mid])))])
                ok=ef<=fov_tol
            if ok:
                best_j=mid;best_curves=(bx,by,bz,br,bf);lo=mid+1
            else:
                hi=mid-1
        bx,by,bz,br,bf=best_curves
        interp=pack((bx,by,bz,br,LINEAR,bf))
        keys.append(dict(frame=best_j,distance=0.,position=pos_vmd[best_j].tolist(),rotation=rot_vmd_rad[best_j].tolist(),
                         fov=int(round(fov_vmd[best_j])),perspective=True,interpolation=interp))
        i=best_j
    return keys


def encode_camera(positions,rotations,fovs,segments,*,compensate=True,source_fps=30,zoom_references=None,max_shift_pmx=None,sparse_reduce=False):
    """Dense or curvature-reduced keys; C1 base translation, optional FOV compensation."""
    factor=source_fps//30
    if source_fps not in (30,60):raise ValueError('Camera samples must be 30 or 60 FPS')
    zrefs=None if zoom_references is None else np.asarray(zoom_references,dtype=float)
    if zrefs is not None and zrefs.shape!=positions.shape:raise ValueError('zoom_references shape must match positions')
    p=positions.copy();r=rotations.copy();f=fovs.copy();count=(len(p)-1)//factor+1
    transitions=[]
    # moveTransition's game blending law is unknown. This is explicitly a
    # compatibility blend, in seconds, constrained to the incoming shot.
    tau=0.15*source_fps
    for j,seg in enumerate(segments):
        a,b=seg['start_vmd_frame']*factor,seg['end_vmd_frame']*factor
        r[a:b+1]=np.unwrap(r[a:b+1],axis=0)
        if not a or seg['hard_cut']:
            r[a:b+1]-=2*np.pi*np.round(r[a]/(2*np.pi))
            continue
        previous_end=a-factor
        r[a:b+1]+=2*np.pi*np.round((r[previous_end]-r[a])/(2*np.pi))
        full_n=max(2,int(round(seg['transition_seconds']*source_fps))+1)
        n=min(b-a+1,full_n)
        if n<2:continue
        u=np.arange(n)/(full_n-1);w=(u*u*(3-2*u))[:,None]
        previous_start=segments[j-1]['start_vmd_frame']*factor
        dt=np.arange(factor,n+factor)[:,None]
        decay=tau*(1.-np.exp(-dt/tau))
        sub_dt=np.arange(1,factor)[:,None]
        sub_decay=tau*(1.-np.exp(-sub_dt/tau))
        for values in (p,r,f[:,None]):
            velocity=(values[previous_end]-values[previous_end-factor])/factor if previous_end-factor>=previous_start else np.zeros_like(values[previous_end])
            predicted=values[previous_end]+decay*velocity
            if factor>1:values[previous_end+1:a]=values[previous_end]+sub_decay*velocity
            values[a:a+n]=(1-w)*predicted+w*values[a:a+n]
        f[:]=np.clip(f,1,179)
        transitions.append(dict(start_frame=a/factor,end_frame=(a+n-1)/factor,requested_seconds=seg['transition_seconds']))
    full_p,full_r,full_f=p,r,f
    p=p[::factor];r=r[::factor];f=f[::factor]
    forward=rotate(euler_unity(-np.rad2deg(r)),[0.,0.,1.])
    integer_fov=np.clip(np.rint(f),1,179).astype(int)
    correction=np.zeros_like(p);depths=np.zeros(count);valid=np.zeros(count,dtype=bool)
    if compensate:
        zrefs_vmd=None if zrefs is None else zrefs[::factor]
        for seg in segments:
            a,b=seg['start_vmd_frame'],seg['end_vmd_frame']
            reference=zrefs_vmd[a:b+1] if zrefs_vmd is not None else np.asarray(seg['zoom_reference'],dtype=float)
            depth=np.sum((reference-p[a:b+1])*forward[a:b+1],axis=1)
            good=depth>1e-4;depths[a:b+1]=depth;valid[a:b+1]=good
            ratio=np.tan(np.deg2rad(f[a:b+1])/2)/np.tan(np.deg2rad(integer_fov[a:b+1])/2)
            raw_shift=np.where(good,depth*(1-ratio),0)
            correction[a:b+1]=forward[a:b+1]*_cap_shift(raw_shift,max_shift_pmx)[:,None]
    output=p+correction
    # Derivatives must never cross a hard cut. Soft joins share a tangent.
    blocks=[];start=0
    for seg in segments:
        if seg['hard_cut']:blocks.append((start,seg['start_vmd_frame']));start=seg['start_vmd_frame']
    blocks.append((start,count))
    cuts={s['start_vmd_frame'] for s in segments if s['hard_cut']}
    if sparse_reduce:
        linear_interp=pack((LINEAR,)*6)
        step_interp=bytes([127,127,0,0]*6)
        keys=[]
        use_full=factor==2 and not compensate
        full_rot_deg=np.rad2deg(full_r) if use_full else None
        for a,b in blocks:
            in_interp=step_interp if a in cuts else linear_interp
            keys.extend(reduce_block_keys(
                output,r,integer_fov.astype(float),a,b-1,in_interp,
                pos_samples=full_p if use_full else None,
                rot_samples_deg=full_rot_deg,
                fov_samples=full_f if use_full else None,
                factor=factor if use_full else 1,
                pos_tol=0.16,rot_tol=0.18,fov_tol=0.75))
        return keys,output,dict(interpolation='shape-preserving 60 FPS sampling reduced to curvature-adaptive cubic VMD Beziers; preserve hard cuts and moveTransition blends',
            source_sample_fps=source_fps,
            transition_policy='approximate cubic smoothstep blend in seconds; original game law unverified',
            transitions=transitions,zoom_compensation=dict(enabled=bool(compensate),
                max_shift_limit_pmx=None if max_shift_pmx is None else float(max_shift_pmx),
                policy='preserve fractional-FOV projection at shot reference depth by shifting optical center along view axis',
                frames_compensated=int(valid.sum()),frames_without_forward_reference=int((~valid).sum()) if compensate else 0,
                max_shift_pmx=float(np.linalg.norm(correction,axis=1).max()),
                limitation='Reference-plane approximation: other scene depths have a small perspective change.'),
            hard_cut_frames=sorted(cuts),sparse_reduced=True,key_count=len(keys))
    ps=np.zeros_like(p);rs=np.zeros_like(r)
    for a,b in blocks:ps[a:b]=slopes(p[a:b]);rs[a:b]=slopes(r[a:b])
    keys=[]
    for i in range(count):
        curves=[LINEAR]*6
        if i and i not in cuts:
            delta=output[i]-output[i-1];dc=correction[i]-correction[i-1]
            curves[:3]=[channel_curve(delta[k],ps[i-1,k]+dc[k],ps[i,k]+dc[k]) for k in range(3)]
            dr=r[i]-r[i-1];norm=float(np.dot(dr,dr))
            if norm>1e-16:
                curves[3]=channel_curve(1.,float(np.dot(rs[i-1],dr)/norm),float(np.dot(rs[i],dr)/norm))
            if factor==2:
                midpoint=2*i-1;fm=full_f[midpoint]
                rm=full_r[midpoint];pm=full_p[midpoint]
                direction=rotate(euler_unity(-np.rad2deg(rm)),[0.,0.,1.])
                ref_mid=zrefs[midpoint] if zrefs is not None else np.asarray(next(s for s in segments if s['start_vmd_frame']<=i<=s['end_vmd_frame'])['zoom_reference'],dtype=float)
                dm=float(np.dot(ref_mid-pm,direction))
                target_fov=fm
                if compensate and dm>1e-4:
                    corrected_depth=dm-float(np.dot((correction[i-1]+correction[i])/2,direction))
                    if corrected_depth>1e-4:target_fov=np.rad2deg(2*np.arctan(dm/corrected_depth*np.tan(np.deg2rad(fm)/2)))
                dg=integer_fov[i]-integer_fov[i-1]
                curves[5]=channel_curve(dg,dg,dg,target_fov-integer_fov[i-1])
                # x1+x2=127, so u=.5 at the center of this VMD interval.
                gf=integer_fov[i-1]+dg*(.125+.375*(curves[5][1]+curves[5][3])/127)
                cm=direction*_cap_shift(dm*(1-np.tan(np.deg2rad(fm)/2)/np.tan(np.deg2rad(gf)/2)),max_shift_pmx) if compensate and dm>1e-4 else np.zeros(3)
                curves[:3]=[channel_curve(delta[k],ps[i-1,k]+dc[k],ps[i,k]+dc[k],pm[k]+cm[k]-output[i-1,k]) for k in range(3)]
                if norm>1e-16:curves[3]=channel_curve(1.,float(np.dot(rs[i-1],dr)/norm),float(np.dot(rs[i],dr)/norm),float(np.dot(rm-r[i-1],dr)/norm))
        interp=bytes([127,127,0,0]*6) if i in cuts else pack(curves)
        keys.append(dict(frame=i,distance=0.,position=output[i].tolist(),rotation=r[i].tolist(),
                         fov=int(integer_fov[i]),perspective=True,interpolation=interp))
    return keys,output,dict(interpolation='dense cubic translation and coupled FOV correction; 60 FPS midpoints constrain 30 FPS VMD curves',source_sample_fps=source_fps,
        transition_policy='approximate quintic blend in seconds, bounded to incoming shot; original game law unverified',
        transitions=transitions,zoom_compensation=dict(enabled=bool(compensate),
            max_shift_limit_pmx=None if max_shift_pmx is None else float(max_shift_pmx),
            policy='preserve fractional-FOV projection at shot reference depth by shifting optical center along view axis',
            frames_compensated=int(valid.sum()),frames_without_forward_reference=int((~valid).sum()) if compensate else 0,
            max_shift_pmx=float(np.linalg.norm(correction,axis=1).max()),
            limitation='Reference-plane approximation: other scene depths have a small perspective change.'),
        hard_cut_frames=sorted(cuts))
