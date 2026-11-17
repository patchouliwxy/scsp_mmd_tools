"""PMXEditor-stage core controls, independent of bone indices and model size."""
import argparse,json,sys,hashlib,shutil
from pathlib import Path
import os
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pmx_io as pmx


def index(model,name):
    found=[i for i,b in enumerate(model.bones) if b.name==name]
    if len(found)!=1:raise ValueError('Expected one bone '+name+': '+str(found))
    return found[0]


def create(model,name,english,position,parent,movable=False):
    if any(b.name==name for b in model.bones):raise ValueError('Already exists: '+name)
    bone=pmx.Bone();bone.name=name;bone.name_e=english;bone.location=list(position);bone.parent=parent;bone.displayConnection=[0,0,-max(0.1,max(v.co[1] for v in model.vertices)*0.025)];bone.isMovable=movable
    model.bones.append(bone);return len(model.bones)-1


def body(model):
    root=index(model,'全ての親');lower=index(model,'下半身');upper=index(model,'上半身')
    center=create(model,'センター','center',model.bones[lower].location,root,True)
    model.bones[lower].parent=center;model.bones[upper].parent=center
    return {'center':center,'upper_lower_parent':'センター','original_pelvis_weights_preserved':True,'coordinates':'Source pelvis pivot in PMX coordinates'}


def eyes(model):
    left=index(model,'左目');right=index(model,'右目');head=index(model,'頭');height=max(v.co[1] for v in model.vertices)-min(v.co[1] for v in model.vertices)
    position=[(a+b)*0.5 for a,b in zip(model.bones[left].location,model.bones[right].location)];position[0]=0;position[2]-=height*0.05
    control=create(model,'両目','eyes',position,head)
    result=[]
    for i in (left,right):
        b=model.bones[i]
        if b.hasAdditionalRotate or b.hasAdditionalLocation:raise ValueError('Existing eye grant needs explicit handling')
        b.hasAdditionalRotate=True;b.hasAdditionalLocation=False;b.additionalTransform=(control,1.0);b.isMovable=False
        result.append({'eye':b.name,'parent_preserved':model.bones[b.parent].name,'grant':'両目','ratio':1.0})
    return result


def order(model):
    visiting=set();cache={}
    def depth(i):
        if i in cache:return cache[i]
        if i in visiting:raise ValueError('Bone dependency cycle at '+model.bones[i].name)
        visiting.add(i);b=model.bones[i];deps=[]
        if b.parent is not None and b.parent>=0:deps.append(b.parent)
        if (b.hasAdditionalRotate or b.hasAdditionalLocation) and b.additionalTransform and b.additionalTransform[0]>=0:deps.append(b.additionalTransform[0])
        value=max((depth(d)+1 for d in deps),default=0);visiting.remove(i);cache[i]=value;return value
    for i,b in enumerate(model.bones):b.transform_order=depth(i)
    # IK targets are evaluated after the effector and link bones.
    for b in model.bones:
        if b.isIK:b.transform_order=max(b.transform_order,max([model.bones[b.target].transform_order]+[model.bones[l.target].transform_order for l in b.ik_links])+1)
    return max(b.transform_order for b in model.bones)


def display(model):
    def group(name,english,indices,special=False,morph=False):
        d=pmx.Display();d.name=name;d.name_e=english;d.isSpecial=special;d.data=[(1 if morph else 0,i) for i in indices];return d
    model.display=[group('Root','Root',[index(model,'全ての親')],True),group('表情','Facial',range(len(model.morphs)),True,True)]
    groups={'センター':[],'体':[],'ＩＫ':[],'目':[],'腕':[],'指':[],'足':[],'補助':[]}
    for i,b in enumerate(model.bones):
        n=b.name
        if n=='全ての親':continue
        if n in ('センター','グルーブ','腰'):g='センター'
        elif 'ＩＫ' in n or 'IK' in n:g='ＩＫ'
        elif n in ('両目','左目','右目'):g='目'
        elif any(t in n for t in ('親指','人指','中指','薬指','小指')):g='指'
        elif n.startswith(('左','右')) and any(t in n for t in ('肩','腕','ひじ','手首','手捩')):g='腕'
        elif n.startswith(('左','右')) and any(t in n for t in ('足','ひざ','つま先')):g='足'
        elif n in ('下半身','上半身','上半身2','上半身3','首','頭'):g='体'
        else:g='補助'
        groups[g].append(i)
    for g,indices in groups.items():
        if indices:model.display.append(group(g,g,indices))


def names_and_panels(model):
    root=Path(__file__).resolve().parents[1];bones=json.loads((root/'Resources/mmd_bone_names.json').read_text(encoding='utf-8'));morphs=json.loads((root/'Resources/unity_morph_names.json').read_text(encoding='utf-8'))
    bone_english={r['japanese']:r['english'] for r in bones['mappings']}
    reverse={v:k for k,v in morphs['source_names'].items()};combos={c['name']:c for c in morphs['combinations']};generated=morphs.get('generated_morphs',{})
    for b in model.bones:
        if not b.name_e:b.name_e=bone_english.get(b.name,b.name)
    for m in model.morphs:
        source=reverse.get(m.name,m.name)
        if m.name in combos:m.category=combos[m.name]['panel'];m.name_e=combos[m.name]['english']
        elif m.name in generated:m.category=generated[m.name]['panel'];m.name_e=generated[m.name]['english']
        elif source.startswith('eyebrow_'):m.category=1
        elif source.startswith(('eye_','eyelash')):m.category=2
        elif source in ('A','I','U','E','O') or source.startswith('mouth_'):m.category=3
        else:m.category=4
        if not m.name_e:m.name_e=morphs['english_names'].get(source,source)


def validate(model):
    names=[b.name for b in model.bones]
    if len(set(names))!=len(names):raise ValueError('Duplicate bone names')
    for i,b in enumerate(model.bones):
        for parent in ([b.parent] if b.parent is not None and b.parent>=0 else []):
            if parent==i or parent>=len(names):raise ValueError('Invalid parent')
        if b.isIK and (b.target is None or not b.ik_links):raise ValueError('Incomplete IK')
    for v in model.vertices:
        if any(i<0 or i>=len(names) for i in v.weight.bones):raise ValueError('Invalid vertex bone index')
    if model.rigids or model.joints:raise ValueError('This pipeline expects no physics')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['body','finalize'],required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);source=Path(a.input).resolve(strict=True);out=Path(a.output).resolve();out.parent.mkdir(exist_ok=True,parents=True)
    if out.exists():raise FileExistsError(out)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest();model=pmx.load(str(source));report={'phase':a.phase,'input':str(source),'input_sha256':source_hash,'before':{'bones':len(model.bones),'vertices':len(model.vertices)}}
    if a.phase=='body':report['changes']=body(model)
    else:
        report['eye_grants']=eyes(model);names_and_panels(model);display(model)
        folder=out.parent/'textures';folder.mkdir(exist_ok=True)
        for i,t in enumerate(model.textures):
            raw=Path(str(t.path).replace('\\','/'))
            src=raw if raw.is_absolute() else (source.parent/raw)
            dest=folder/(str(i).zfill(2)+'_'+src.name)
            if not src.is_file():raise FileNotFoundError(src)
            shutil.copy2(src,dest);t.path=f'textures\\{dest.name}'
    report['max_transform_layer']=order(model);validate(model)
    pmx.save(str(out),model,add_uv_count=max((len(v.additional_uvs) for v in model.vertices),default=0))
    report['after']={'bones':len(model.bones),'vertices':len(model.vertices),'faces':len(model.faces),'morphs':len(model.morphs),'IK':[b.name for b in model.bones if b.isIK],'frames':[d.name for d in model.display]};report['output']=str(out);report['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise AssertionError('Input changed')
    out.with_suffix('.operation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('PE_CORE',json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
