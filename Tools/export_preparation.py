"""Make the deliverable export-safe by excluding archived source/reference objects."""
import json
import bpy


def prepare(root,rig,meshes):
    main=list(meshes)
    if not main or any(o.type!='MESH' or not o.get('scsp_merged_part') for o in main):raise ValueError('Export scope requires merged main parts')
    protected=set(main)|{root,rig}
    references=set()
    collections=[c for c in bpy.data.collections if c.name.startswith(('SCSP_HiddenMeshReferences','SCSP_SourceReferences'))]
    for collection in collections:references.update(collection.all_objects)
    references.update(o for o in bpy.data.objects if o.get('scsp_reference_only'))
    if references&protected:raise ValueError('A required model object is in a reference collection')
    unexpected=[o.name for o in bpy.data.objects if o.type=='MESH' and o not in set(main)|references]
    if unexpected:raise ValueError('Unclassified meshes need an explicit export decision: '+str(unexpected))
    removed=[]
    for obj in sorted(references,key=lambda o:o.name):
        record={'object':obj.name,'type':obj.type,'vertices':len(obj.data.vertices) if obj.type=='MESH' else 0,'source_object':obj.get('scsp_source_object')}
        if obj.type=='MESH':record['materials']=[m.name if m else None for m in obj.data.materials]
        removed.append(record);bpy.data.objects.remove(obj,do_unlink=True)
    for collection in collections:
        if collection.name in bpy.data.collections:bpy.data.collections.remove(collection)
    actual=[o for o in bpy.data.objects if o.type=='MESH']
    if set(actual)!=set(main):raise AssertionError('Exportable mesh scope is not exactly the main parts')
    report={'module':'export_preparation','main_meshes':[o.name for o in main],'main_vertices':sum(len(o.data.vertices) for o in main),'removed_objects':removed,'removed_reference_vertices':sum(r['vertices'] for r in removed),'archive_policy':'References remain in the preceding working blend/intermediates; no hidden reference meshes in this export-ready copy','export_ui_recommendation':'MMD Tools: Visible Meshes Only enabled as additional filter','no_mesh_geometry_or_shape_changes':True}
    root['scsp_export_scope']=json.dumps(report,ensure_ascii=False)
    return report
