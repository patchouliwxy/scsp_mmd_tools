"""Pure PMX control-rig stages, matching inspected installed-plugin rules."""
import argparse,json,math,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mathutils import Vector
from pe_core import pmx,index,create,order,validate


def influences(v):
    w=v.weight
    if w.type==0:values=[1.0]
    elif w.type==1:values=[w.weights[0],1-w.weights[0]]
    elif w.type==3:values=[w.weights.weight,1-w.weights.weight]
    else:values=list(w.weights)
    return [(b,float(a)) for b,a in zip(w.bones,values)]


def _ensure_knee_prebend(model, leg, knee, ankle, ratio=0.02):
    """Give the knee a small forward pre-bend (-Z in PMX) and link tails to children.

    Without a forward pre-bend, backward+downward センター shifts trap Blender/MMD IK
    on the 0-degree knee extension limit, locking the leg straight instead of bending.
    """
    leg_bone = model.bones[leg]
    knee_bone = model.bones[knee]
    ankle_bone = model.bones[ankle]
    leg_loc = leg_bone.location
    knee_loc = list(knee_bone.location)
    ankle_loc = ankle_bone.location
    dy = ankle_loc[1] - leg_loc[1]
    t = (knee_loc[1] - leg_loc[1]) / dy if abs(dy) > 1e-6 else 0.5
    chord_z = leg_loc[2] + t * (ankle_loc[2] - leg_loc[2])
    thigh_len = math.sqrt(sum((a - b) ** 2 for a, b in zip(knee_loc, leg_loc)))
    target_z = chord_z - thigh_len * ratio
    applied = 0.0
    if knee_loc[2] > target_z:
        applied = knee_loc[2] - target_z
        knee_loc[2] = target_z
        knee_bone.location = knee_loc
    leg_bone.displayConnection = knee
    knee_bone.displayConnection = ankle
    return float(applied)


def leg_ik(model):
    root=index(model,'全ての親');result=[]
    for side in ('左','右'):
        leg=index(model,side+'足');knee=index(model,side+'ひざ');ankle=index(model,side+'足首');toe=index(model,side+'つま先')
        prebend = _ensure_knee_prebend(model, leg, knee, ankle)
        control=create(model,side+'足ＩＫ','leg IK_'+('L' if side=='左' else 'R'),model.bones[ankle].location,root,True)
        b=model.bones[control];b.isIK=True;b.target=ankle;b.loopCount=40;b.rotationConstraint=math.radians(114.5916);b.displayConnection=[0,0,1]
        link=pmx.IKLink();link.target=knee;link.minimumAngle=[-math.pi,0,0];link.maximumAngle=[math.radians(-0.5),0,0]
        upper=pmx.IKLink();upper.target=leg;b.ik_links=[link,upper]
        tip=create(model,side+'つま先ＩＫ','toe IK_'+('L' if side=='左' else 'R'),model.bones[toe].location,control,True)
        b=model.bones[tip];b.isIK=True;b.target=toe;b.loopCount=3;b.rotationConstraint=math.radians(229.18311);b.displayConnection=[0,-0.7,0]
        link=pmx.IKLink();link.target=ankle;b.ik_links=[link]
        result.append({'side':side,'leg_control':control,'toe_control':tip,'knee_limit_degrees':[-180,-0.5],'knee_prebend_pmx':prebend})
    return result


def _encode_weights(pairs):
    merged={}
    for bone_idx,weight in pairs:
        if weight>1e-7:merged[bone_idx]=merged.get(bone_idx,0.0)+float(weight)
    if not merged:raise ValueError('Vertex lost all bone weights during twist redistribution')
    items=sorted(merged.items(),key=lambda item:item[1],reverse=True)[:4]
    total=sum(w for _,w in items)
    items=[(b,w/total) for b,w in items]
    bw=pmx.BoneWeight()
    if len(items)==1:
        bw.type=pmx.BoneWeight.BDEF1;bw.bones=[items[0][0]];bw.weights=[]
    elif len(items)==2:
        bw.type=pmx.BoneWeight.BDEF2;bw.bones=[items[0][0],items[1][0]];bw.weights=[items[0][1]]
    else:
        while len(items)<4:items.append((0,0.0))
        bw.type=pmx.BoneWeight.BDEF4;bw.bones=[b for b,_ in items];bw.weights=[w for _,w in items]
    return bw


def _twist_local_coordinate(axis):
    y_bl=Vector((axis.x,axis.z,axis.y)).normalized()
    ref_z=Vector((0.0,-1.0,0.0)) if abs(y_bl.y)<0.95 else Vector((0.0,0.0,1.0))
    x_bl=y_bl.cross(ref_z).normalized()
    z_bl=x_bl.cross(y_bl).normalized()
    return pmx.Coordinate([x_bl.x,x_bl.z,x_bl.y],[z_bl.x,z_bl.z,z_bl.y])


def twist(model,start_name,end_name,control_name,english):
    start=index(model,start_name);end=index(model,end_name);a=Vector(model.bones[start].location);b=Vector(model.bones[end].location);axis=(b-a).normalized();length=(b-a).length
    if length<1e-7:raise ValueError('Degenerate twist span')
    u_low,u_high=(0.18,0.92) if start_name.endswith('腕') else (0.18,0.90)
    low_u=u_low*length;high_u=u_high*length;pivot=a.lerp(b,0.5)
    coord=_twist_local_coordinate(axis);disp=list(axis*max(length*0.15,0.05))
    control=create(model,control_name,english,pivot,start);bone=model.bones[control];bone.axis=list(axis);bone.displayConnection=list(disp);bone.localCoordinate=pmx.Coordinate(list(coord.x_axis),list(coord.z_axis))
    chain=[start]
    for j in range(1,4):
        pos=a+axis*(low_u+(high_u-low_u)*j/4)
        n=create(model,control_name+str(j),english+' '+str(j),pos,start)
        child=model.bones[n];child.hasAdditionalRotate=True;child.additionalTransform=(control,j/4);child.visible=False;child.isMovable=False;child.displayConnection=list(disp);child.localCoordinate=pmx.Coordinate(list(coord.x_axis),list(coord.z_axis));chain.append(n)
    chain.append(control)
    smooth_vertices=0;clamped_vertices=0
    for v in model.vertices:
        weights=influences(v);start_weight=sum(w for bone_idx,w in weights if bone_idx==start)
        if start_weight<=1e-7:continue
        others=[(bone_idx,w) for bone_idx,w in weights if bone_idx!=start and w>1e-7]
        s=(Vector(v.co)-a).dot(axis)
        t=max(0.0,min(1.0,(s-low_u)/(high_u-low_u)))
        frac=t*t*(3.0-2.0*t)
        q=frac*4.0
        if frac<=0.0:
            replaced=[(start,start_weight)]
        elif frac>=1.0:
            replaced=[(control,start_weight)]
        elif len(others)<=2:
            j=min(3,int(q));alpha=q-j
            replaced=[]
            if 1.0-alpha>1e-6:replaced.append((chain[j],start_weight*(1.0-alpha)))
            if alpha>1e-6:replaced.append((chain[j+1],start_weight*alpha))
        else:
            replaced=[(chain[min(4,round(q))],start_weight)];clamped_vertices+=1
        v.weight=_encode_weights(others+replaced);smooth_vertices+=1
    model.bones[end].parent=control
    return {'control':control_name,'parent':start_name,'end_reparented':end_name,'main_axis':list(axis),'ratio_helpers':[0.25,0.5,0.75],'u_span':[u_low,u_high],'redistributed_vertices':smooth_vertices,'clamped_4bone_vertices':clamped_vertices,'projection_range':[low_u-(pivot-a).dot(axis),high_u-(pivot-a).dot(axis)],'position_or_geometry_changed':False,'algorithm':'Smoothstep BDEF4 start-weight redistribution along segment span [u_low,u_high] with aligned helper axes'}


def semi(model):
    records=[]
    for side,suffix in [('左','L'),('右','R')]:records.append(twist(model,side+'腕',side+'ひじ',side+'腕捩','arm twist_'+suffix))
    for side,suffix in [('左','L'),('右','R')]:records.append(twist(model,side+'ひじ',side+'手首',side+'手捩','wrist twist_'+suffix))
    center=index(model,'センター');location=list(model.bones[center].location);location[1]+=0.2
    groove=create(model,'グルーブ','groove',location,center,True);model.bones[groove].displayConnection=[0,1.4,0]
    for i,b in enumerate(model.bones):
        if i!=groove and b.parent==center and b.name!='センター先':b.parent=groove
    feet=[];shoulders=[]
    for side,suffix in [('左','L'),('右','R')]:
        foot=index(model,side+'足ＩＫ');location=list(model.bones[foot].location);location[1]=model.bones[index(model,'全ての親')].location[1]
        parent=create(model,side+'足IK親','leg IKP_'+suffix,location,model.bones[foot].parent,True);model.bones[parent].displayConnection=foot;model.bones[foot].parent=parent;feet.append(side+'足IK親')
        shoulder=index(model,side+'肩');arm=index(model,side+'腕')
        p=create(model,side+'肩P','shoulderP_'+suffix,model.bones[shoulder].location,model.bones[shoulder].parent)
        c=create(model,side+'肩C','shoulderC_'+suffix,model.bones[arm].location,shoulder)
        model.bones[c].hasAdditionalRotate=True;model.bones[c].additionalTransform=(p,-1.0);model.bones[c].visible=False
        model.bones[shoulder].parent=p;model.bones[arm].parent=c;shoulders.extend([side+'肩P',side+'肩C'])
    return {'twist':records,'groove':'グルーブ','ik_parents':feet,'shoulder_controls':shoulders,'method':'Pure PMX implementation of inspected plugin data rules; original GUI DLL Run not called'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['leg','semi'],required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(a.input).resolve(strict=True);out=Path(a.output).resolve();out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():raise FileExistsError(out)
    sha=hashlib.sha256(source.read_bytes()).hexdigest();m=pmx.load(str(source));report={'phase':a.phase,'input':str(source),'input_sha256':sha,'before_bones':len(m.bones),'implementation':'Background PMX data script, matching installed plugin parameters, not a DLL execution claim'}
    report['changes']=leg_ik(m) if a.phase=='leg' else semi(m)
    order(m);validate(m);pmx.save(str(out),m,add_uv_count=max((len(v.additional_uvs) for v in m.vertices),default=0));report['after_bones']=len(m.bones);report['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    if sha!=hashlib.sha256(source.read_bytes()).hexdigest():raise AssertionError('Input changed')
    out.with_suffix('.operation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('PE_CONTROLS',json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
