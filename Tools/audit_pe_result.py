"""Structural/data audit of the delivered PMX; no motion simulation or renders."""
import argparse,json,sys,hashlib,math
from pathlib import Path
import os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pmx_io as pmx

def signature(model):
    rows=[(tuple(v.co),tuple(v.normal),tuple(v.uv),tuple(tuple(x) for x in v.additional_uvs)) for v in model.vertices]
    return hashlib.sha256(repr((rows,model.faces)).encode()).hexdigest()

def weights(model):
    result=[]
    for v in model.vertices:
        w=v.weight
        if w.type==0:values=[1.0]
        elif w.type==1:values=[float(w.weights[0]),1-float(w.weights[0])]
        elif w.type==3:values=[float(w.weights.weight),1-float(w.weights.weight)]
        else:values=list(w.weights)
        result.append([(model.bones[i].name,float(value)) for i,value in zip(w.bones,values) if value>1e-7])
    return result

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);old=pmx.load(str(Path(a.input).resolve()));new=pmx.load(str(Path(a.output).resolve()))
oldw=weights(old);neww=weights(new);names=[b.name for b in new.bones]
changed=[i for i,(u,v) in enumerate(zip(oldw,neww)) if u!=v]
required=['全ての親','センター','グルーブ','両目','左足ＩＫ','右足ＩＫ','左つま先ＩＫ','右つま先ＩＫ','左足IK親','右足IK親','左腕捩','右腕捩','左手捩','右手捩','左肩P','右肩P','左肩C','右肩C']
r={'input':str(Path(a.input).resolve()),'output':str(Path(a.output).resolve()),'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),'vertices':len(new.vertices),'faces':len(new.faces),'bones':len(new.bones),'morphs':len(new.morphs),'rigids':len(new.rigids),'joints':len(new.joints),'geometry_normals_uv_faces_preserved':signature(old)==signature(new),'changed_weight_vertices':len(changed),'required_missing':[n for n in required if n not in names],'duplicate_names':len(names)-len(set(names)),'textures_missing':[t.path for t in new.textures if not Path(t.path).is_file()],'max_weight_sum_error':max(abs(sum(v for _,v in row)-1) for row in neww),'controls':[{'name':b.name,'parent':new.bones[b.parent].name if b.parent is not None and b.parent>=0 else None,'order':b.transform_order,'grant':(new.bones[b.additionalTransform[0]].name,b.additionalTransform[1]) if b.additionalTransform else None,'axis':b.axis,'IK_target':new.bones[b.target].name if b.isIK else None,'links':[new.bones[l.target].name for l in b.ik_links] if b.isIK else []} for b in new.bones if b.name in required],'manual_review':'No motion/render/effect validation performed; user review required'}
if not r['geometry_normals_uv_faces_preserved'] or r['required_missing'] or r['duplicate_names'] or r['textures_missing'] or r['max_weight_sum_error']>1e-4 or r['rigids'] or r['joints']:raise AssertionError(json.dumps(r,ensure_ascii=False))
Path(a.report).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');print('STRUCTURAL_AUDIT',json.dumps({k:v for k,v in r.items() if k!='controls'},ensure_ascii=False))
