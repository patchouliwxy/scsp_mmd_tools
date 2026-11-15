"""Eye module: bindings, planar iris, retreat, sockets, rotation pivots."""
from preprocess_parts import prepare_eye_bindings
from eye_sockets import EyeSocketError,enforce_eye_bindings,build_eye_sockets,preflight
from eye_rotation_pivots import configure


def prepare(parts,config):
    socket_options=config.get('eye_sockets')
    preserve_source=config.get('eye_geometry_fallback')=='preserve_source'
    skipped_geometry=None
    try:
        socket_preflight=preflight(parts,socket_options)
    except EyeSocketError as exc:
        if not preserve_source:
            raise
        reason=str(exc)
        skipped_geometry={
            'schema':2,
            'operation':'eye_socket_geometry',
            'status':'skipped_unsupported',
            'fallback':'preserve_source',
            'reason':reason,
            'warnings':[
                'Eye socket geometry preflight unsupported; preserved source '
                f'eye geometry and continued eye weight constraints: {reason}'
            ],
            'preflight_was_read_only':True,
            'source_geometry_modified':False,
            'socket_count':0,
            'iris_flattened':False,
            'iris_retreated':False,
            'skipped_operations':['iris_flatten','iris_retreat','eye_socket_creation'],
        }
        socket_preflight=None

    result={'pending_follow_up':None,
            'warnings':[],
            'geometry_preflight':socket_preflight,
            'initial_iris_bindings':prepare_eye_bindings(parts),
            'weight_policy':enforce_eye_bindings(parts)}
    if skipped_geometry is not None:
        skipped_geometry['weight_bindings_completed']=True
        result['geometry']=skipped_geometry
        result['warnings'].extend(skipped_geometry['warnings'])
    else:
        geometry=build_eye_sockets(parts,socket_options)
        geometry['status']='ok'
        result['geometry']=geometry
    from import_processing import normalize_roughness
    result['generated_material_roughness']=normalize_roughness([o for part in parts for o in part['meshes']])
    return result


def finalize_rig(rig,meshes):
    # The unified skeleton must exist before its eye pivots can be positioned.
    return configure(rig,meshes)
