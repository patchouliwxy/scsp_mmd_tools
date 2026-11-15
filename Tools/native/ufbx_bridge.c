/*
 * Clean-room C ABI bridge wrapping MIT ufbx for Blender headless/runtime FBX import.
 * Build on Windows x64 (MinGW):
 *   gcc -O2 -shared -static-libgcc -I../../ThirdParty/ufbx ../../ThirdParty/ufbx/ufbx.c ufbx_bridge.c -o ../ufbx_bridge.dll
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ufbx.h"

#ifdef _WIN32
#define SCSP_API __declspec(dllexport)
#else
#define SCSP_API
#endif

typedef struct ScspUvLayer {
    const char *name;
    float *loop_uvs; /* num_kept_loops * 2 */
} ScspUvLayer;

typedef struct ScspShapeData {
    const char *name;
    int32_t num_offsets;
    int32_t *vert_indices;
    float *deltas_xyz;
} ScspShapeData;

typedef struct ScspMeshData {
    int32_t node_index;
    const char *name;
    int32_t num_verts;
    int32_t num_polys;
    int32_t num_loops;
    int32_t is_skinned;
    int32_t has_normals;
    float *verts_xyz;          /* num_verts * 3 */
    int32_t *loop_start;       /* num_polys */
    int32_t *loop_total;       /* num_polys */
    int32_t *poly_mat_index;   /* num_polys */
    int32_t *loop_verts;       /* num_loops */
    float *loop_normals;       /* num_loops * 3 */
    int32_t num_uv_layers;
    ScspUvLayer *uv_layers;
    int32_t num_mats;
    int32_t *scene_mat_indices;
    int32_t num_skin_weights;
    int32_t *skin_bone_idx;
    int32_t *skin_vert_idx;
    float *skin_weights;
    int32_t num_shapes;
    ScspShapeData *shapes;
} ScspMeshData;

typedef struct ScspBoneData {
    int32_t node_index;
    const char *name;
    int32_t parent_bone_idx;
    float bind_world[16]; /* row-major 4x4 */
    float tail_len;
} ScspBoneData;

typedef struct ScspNodeInfo {
    const char *name;
    int32_t kind;        /* 0=empty, 1=bone, 2=mesh, 3=armature_root */
    int32_t parent_idx;  /* index into ScspNodeInfo[] or -1 */
    int32_t mesh_idx;
    int32_t bone_idx;
    float world_mat[16]; /* row-major 4x4 */
} ScspNodeInfo;

typedef struct ScspMatInfo {
    const char *name;
    float diffuse_rgba[4];
    const char *tex_filename;
    const char *tex_rel_filename;
} ScspMatInfo;

typedef struct ScspFbxScene {
    ufbx_scene *scene;
    const char *armature_name;
    int32_t armature_node_idx;

    int32_t num_nodes;
    ScspNodeInfo *nodes;
    int32_t *ufbx_to_scsp_node;

    int32_t num_bones;
    ScspBoneData *bones;

    int32_t num_meshes;
    ScspMeshData *meshes;

    int32_t num_mats;
    ScspMatInfo *mats;
} ScspFbxScene;

static void write_mat4_row(float dst[16], const ufbx_matrix *m) {
    dst[0] = (float)m->cols[0].x; dst[1] = (float)m->cols[1].x; dst[2] = (float)m->cols[2].x; dst[3] = (float)m->cols[3].x;
    dst[4] = (float)m->cols[0].y; dst[5] = (float)m->cols[1].y; dst[6] = (float)m->cols[2].y; dst[7] = (float)m->cols[3].y;
    dst[8] = (float)m->cols[0].z; dst[9] = (float)m->cols[1].z; dst[10] = (float)m->cols[2].z; dst[11] = (float)m->cols[3].z;
    dst[12] = 0.0f; dst[13] = 0.0f; dst[14] = 0.0f; dst[15] = 1.0f;
}

static float vec3_dist(ufbx_vec3 a, ufbx_vec3 b) {
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return (float)sqrt(dx * dx + dy * dy + dz * dz);
}

static ufbx_vec3 mk_vec3(double x, double y, double z) {
    ufbx_vec3 v = {x, y, z};
    return v;
}

static ufbx_vec3 vec3_norm_safe(ufbx_vec3 v, ufbx_vec3 fallback) {
    double len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8) return fallback;
    return mk_vec3(v.x / len, v.y / len, v.z / len);
}

static uint64_t tri_key3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t t[3] = {a, b, c};
    if (t[0] > t[1]) { uint32_t x = t[0]; t[0] = t[1]; t[1] = x; }
    if (t[1] > t[2]) { uint32_t x = t[1]; t[1] = t[2]; t[2] = x; }
    if (t[0] > t[1]) { uint32_t x = t[0]; t[0] = t[1]; t[1] = x; }
    /* FNV-1a 64-bit over sorted 3 vertex indices */
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < 3; i++) {
        h ^= t[i];
        h *= 1099511628211ULL;
    }
    return h;
}

typedef struct TriSetSlot {
    uint32_t v[3];
    uint8_t used;
} TriSetSlot;

static int insert_unique_face(TriSetSlot *table, size_t mask, const uint32_t *verts, size_t n) {
    if (n != 3) return 1; /* Keep non-triangles unconditionally */
    uint32_t s[3] = {verts[0], verts[1], verts[2]};
    if (s[0] == s[1] || s[1] == s[2] || s[0] == s[2]) return 0; /* Degenerate */
    if (s[0] > s[1]) { uint32_t x = s[0]; s[0] = s[1]; s[1] = x; }
    if (s[1] > s[2]) { uint32_t x = s[1]; s[1] = s[2]; s[2] = x; }
    if (s[0] > s[1]) { uint32_t x = s[0]; s[0] = s[1]; s[1] = x; }
    size_t idx = (size_t)tri_key3(s[0], s[1], s[2]) & mask;
    while (table[idx].used) {
        if (table[idx].v[0] == s[0] && table[idx].v[1] == s[1] && table[idx].v[2] == s[2]) {
            return 0; /* Duplicate sorted face */
        }
        idx = (idx + 1) & mask;
    }
    table[idx].used = 1;
    table[idx].v[0] = s[0];
    table[idx].v[1] = s[1];
    table[idx].v[2] = s[2];
    return 1;
}

static void extract_dom_uv_layer(
    ufbx_dom_node *dom_elem,
    const ufbx_mesh *mesh,
    const int32_t *kept_orig_corners,
    int32_t num_kept_loops,
    ScspUvLayer **out_layers,
    int32_t *out_count
) {
    if (!dom_elem) return;
    for (size_t ci = 0; ci < dom_elem->children.count; ci++) {
        ufbx_dom_node *ch = dom_elem->children.data[ci];
        if (strcmp(ch->name.data, "LayerElementNormalMapUV") != 0) continue;

        ufbx_dom_node *nm = ufbx_dom_find(ch, "Name");
        ufbx_dom_node *map = ufbx_dom_find(ch, "MappingInformationType");
        ufbx_dom_node *ref = ufbx_dom_find(ch, "ReferenceInformationType");
        ufbx_dom_node *uv_node = ufbx_dom_find(ch, "UV");
        ufbx_dom_node *idx_node = ufbx_dom_find(ch, "UVIndex");
        if (!uv_node || uv_node->values.count == 0) continue;

        const char *uv_name = (nm && nm->values.count > 0 && nm->values.data[0].type == UFBX_DOM_VALUE_STRING)
            ? nm->values.data[0].value_str.data : "UV1";
        const char *map_type = (map && map->values.count > 0) ? map->values.data[0].value_str.data : "";
        const char *ref_type = (ref && ref->values.count > 0) ? ref->values.data[0].value_str.data : "";

        ufbx_dom_value uv_val = uv_node->values.data[0];
        size_t num_uv_pts = 0;
        const double *d_arr = NULL;
        const float *f_arr = NULL;
        if (uv_val.type == UFBX_DOM_VALUE_ARRAY_F64) {
            d_arr = (const double *)uv_val.value_blob.data;
            num_uv_pts = uv_val.value_blob.size / (sizeof(double) * 2);
        } else if (uv_val.type == UFBX_DOM_VALUE_ARRAY_F32) {
            f_arr = (const float *)uv_val.value_blob.data;
            num_uv_pts = uv_val.value_blob.size / (sizeof(float) * 2);
        } else {
            continue;
        }

        const int32_t *idx_arr = NULL;
        size_t num_idx = 0;
        if (idx_node && idx_node->values.count > 0 && idx_node->values.data[0].type == UFBX_DOM_VALUE_ARRAY_I32) {
            idx_arr = (const int32_t *)idx_node->values.data[0].value_blob.data;
            num_idx = idx_node->values.data[0].value_blob.size / sizeof(int32_t);
        }

        int by_vert = (strcmp(map_type, "ByVertice") == 0);
        int indexed = (strcmp(ref_type, "IndexToDirect") == 0);

        float *buf = (float *)calloc((size_t)num_kept_loops * 2, sizeof(float));
        for (int32_t l = 0; l < num_kept_loops; l++) {
            int32_t orig_c = kept_orig_corners[l];
            size_t elem_ix = by_vert ? (size_t)mesh->vertex_indices.data[orig_c] : (size_t)orig_c;
            size_t uv_ix = elem_ix;
            if (indexed && idx_arr && elem_ix < num_idx) {
                uv_ix = (size_t)idx_arr[elem_ix];
            }
            if (uv_ix < num_uv_pts) {
                buf[l * 2 + 0] = d_arr ? (float)d_arr[uv_ix * 2 + 0] : f_arr[uv_ix * 2 + 0];
                buf[l * 2 + 1] = d_arr ? (float)d_arr[uv_ix * 2 + 1] : f_arr[uv_ix * 2 + 1];
            }
        }

        int32_t nc = *out_count;
        *out_layers = (ScspUvLayer *)realloc(*out_layers, (size_t)(nc + 1) * sizeof(ScspUvLayer));
        (*out_layers)[nc].name = uv_name;
        (*out_layers)[nc].loop_uvs = buf;
        *out_count = nc + 1;
    }
}

SCSP_API void *scsp_fbx_load(const char *path_utf8, char *err_buf, int32_t err_cap) {
    if (err_buf && err_cap > 0) err_buf[0] = '\0';
    ufbx_load_opts opts = {0};
    opts.target_axes = ufbx_axes_right_handed_z_up;
    opts.target_unit_meters = 0;
    opts.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
    opts.retain_dom = true;

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(path_utf8, &opts, &error);
    if (!scene) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, (size_t)err_cap, "%s", error.description.data);
        }
        return NULL;
    }

    ScspFbxScene *ctx = (ScspFbxScene *)calloc(1, sizeof(ScspFbxScene));
    ctx->scene = scene;
    ctx->armature_node_idx = -1;
    ctx->armature_name = "Armature";

    /* 1. Collect all non-root scene nodes and map ufbx node typed_id -> node_index */
    int32_t total_nodes = (int32_t)scene->nodes.count;
    ctx->ufbx_to_scsp_node = (int32_t *)malloc((size_t)total_nodes * sizeof(int32_t));
    for (int32_t i = 0; i < total_nodes; i++) ctx->ufbx_to_scsp_node[i] = -1;

    ctx->nodes = (ScspNodeInfo *)calloc((size_t)total_nodes, sizeof(ScspNodeInfo));
    ctx->bones = (ScspBoneData *)calloc((size_t)total_nodes, sizeof(ScspBoneData));
    ctx->meshes = (ScspMeshData *)calloc((size_t)total_nodes, sizeof(ScspMeshData));

    for (size_t i = 0; i < scene->nodes.count; i++) {
        ufbx_node *nd = scene->nodes.data[i];
        if (nd->is_root) continue;
        int32_t ni = ctx->num_nodes++;
        ctx->ufbx_to_scsp_node[nd->typed_id] = ni;
        ScspNodeInfo *info = &ctx->nodes[ni];
        info->name = nd->name.data;
        info->mesh_idx = -1;
        info->bone_idx = -1;
        write_mat4_row(info->world_mat, &nd->node_to_world);
        if (nd->bone) {
            info->kind = 1;
            int32_t bi = ctx->num_bones++;
            info->bone_idx = bi;
            ctx->bones[bi].node_index = ni;
            ctx->bones[bi].name = nd->name.data;
            ctx->bones[bi].parent_bone_idx = -1;
            write_mat4_row(ctx->bones[bi].bind_world, &nd->node_to_world);
            ctx->bones[bi].tail_len = 0.05f;
        } else if (nd->mesh) {
            info->kind = 2;
        } else {
            info->kind = 0;
        }
    }

    /* Resolve parent indices + armature container node (parent of top-most bone) */
    for (size_t i = 0; i < scene->nodes.count; i++) {
        ufbx_node *nd = scene->nodes.data[i];
        if (nd->is_root) continue;
        int32_t ni = ctx->ufbx_to_scsp_node[nd->typed_id];
        int32_t p_ni = (nd->parent && !nd->parent->is_root) ? ctx->ufbx_to_scsp_node[nd->parent->typed_id] : -1;
        ctx->nodes[ni].parent_idx = p_ni;
        if (ctx->nodes[ni].kind == 1) {
            int32_t bi = ctx->nodes[ni].bone_idx;
            if (p_ni >= 0 && ctx->nodes[p_ni].kind == 1) {
                ctx->bones[bi].parent_bone_idx = ctx->nodes[p_ni].bone_idx;
            } else if (ctx->armature_node_idx < 0 && p_ni >= 0) {
                ctx->armature_node_idx = p_ni;
                ctx->armature_name = ctx->nodes[p_ni].name;
                ctx->nodes[p_ni].kind = 3;
            }
        }
    }

    /* Override bone bind_world from first coincident skin cluster bind_to_world */
    uint8_t *has_bind = (uint8_t *)calloc((size_t)ctx->num_bones + 1, 1);
    for (size_t ci = 0; ci < scene->skin_clusters.count; ci++) {
        ufbx_skin_cluster *cl = scene->skin_clusters.data[ci];
        if (!cl->bone_node) continue;
        int32_t ni = ctx->ufbx_to_scsp_node[cl->bone_node->typed_id];
        if (ni < 0 || ctx->nodes[ni].bone_idx < 0) continue;
        int32_t bi = ctx->nodes[ni].bone_idx;
        if (!has_bind[bi] &&
            vec3_dist(cl->bind_to_world.cols[3], cl->bone_node->node_to_world.cols[3]) < 1e-3f) {
            write_mat4_row(ctx->bones[bi].bind_world, &cl->bind_to_world);
            has_bind[bi] = 1;
        }
    }
    free(has_bind);

    /* Compute sensible bone tail length from first non-coincident child bone */
    for (int32_t bi = 0; bi < ctx->num_bones; bi++) {
        float bx = ctx->bones[bi].bind_world[3];
        float by = ctx->bones[bi].bind_world[7];
        float bz = ctx->bones[bi].bind_world[11];
        float best = -1.0f;
        for (int32_t cj = 0; cj < ctx->num_bones; cj++) {
            if (ctx->bones[cj].parent_bone_idx != bi) continue;
            float cx = ctx->bones[cj].bind_world[3] - bx;
            float cy = ctx->bones[cj].bind_world[7] - by;
            float cz = ctx->bones[cj].bind_world[11] - bz;
            float d = sqrtf(cx * cx + cy * cy + cz * cz);
            if (d > 1e-4f) {
                best = d;
                break;
            }
        }
        if (best > 1e-4f) {
            ctx->bones[bi].tail_len = best;
        } else if (ctx->bones[bi].parent_bone_idx >= 0) {
            ctx->bones[bi].tail_len = fmaxf(0.03f, ctx->bones[ctx->bones[bi].parent_bone_idx].tail_len * 0.5f);
        } else {
            ctx->bones[bi].tail_len = 0.05f;
        }
    }

    /* 2. Collect scene materials */
    ctx->num_mats = (int32_t)scene->materials.count;
    ctx->mats = (ScspMatInfo *)calloc((size_t)ctx->num_mats + 1, sizeof(ScspMatInfo));
    for (int32_t mi = 0; mi < ctx->num_mats; mi++) {
        ufbx_material *mat = scene->materials.data[mi];
        ScspMatInfo *out = &ctx->mats[mi];
        out->name = mat->name.data;
        if (mat->fbx.diffuse_color.has_value) {
            out->diffuse_rgba[0] = (float)mat->fbx.diffuse_color.value_vec3.x;
            out->diffuse_rgba[1] = (float)mat->fbx.diffuse_color.value_vec3.y;
            out->diffuse_rgba[2] = (float)mat->fbx.diffuse_color.value_vec3.z;
        } else {
            out->diffuse_rgba[0] = 1.0f; out->diffuse_rgba[1] = 1.0f; out->diffuse_rgba[2] = 1.0f;
        }
        out->diffuse_rgba[3] = 1.0f;
        if (mat->fbx.diffuse_color.texture) {
            out->tex_filename = mat->fbx.diffuse_color.texture->filename.data;
            out->tex_rel_filename = mat->fbx.diffuse_color.texture->relative_filename.data;
        }
    }

    /* 3. Build mesh data */
    for (size_t i = 0; i < scene->nodes.count; i++) {
        ufbx_node *nd = scene->nodes.data[i];
        if (!nd->mesh || nd->is_root) continue;
        int32_t ni = ctx->ufbx_to_scsp_node[nd->typed_id];
        int32_t mi = ctx->num_meshes++;
        ctx->nodes[ni].mesh_idx = mi;
        ScspMeshData *md = &ctx->meshes[mi];
        const ufbx_mesh *m = nd->mesh;
        md->node_index = ni;
        md->name = nd->name.data;
        md->num_verts = (int32_t)m->num_vertices;
        md->is_skinned = (m->skin_deformers.count > 0) ? 1 : 0;
        md->has_normals = m->vertex_normal.exists ? 1 : 0;

        /* Positions transformed by geometry_to_node */
        md->verts_xyz = (float *)malloc((size_t)md->num_verts * 3 * sizeof(float));
        for (int32_t v = 0; v < md->num_verts; v++) {
            ufbx_vec3 p = ufbx_transform_position(&nd->geometry_to_node, m->vertices.data[v]);
            md->verts_xyz[v * 3 + 0] = (float)p.x;
            md->verts_xyz[v * 3 + 1] = (float)p.y;
            md->verts_xyz[v * 3 + 2] = (float)p.z;
        }

        /* Filter duplicate sorted-vertex polygons */
        size_t cap = 16;
        while (cap < m->num_faces * 2) cap <<= 1;
        TriSetSlot *tri_set = (TriSetSlot *)calloc(cap, sizeof(TriSetSlot));
        int32_t *kept_face_indices = (int32_t *)malloc((size_t)m->num_faces * sizeof(int32_t));
        int32_t kept_polys = 0;
        int32_t kept_loops = 0;
        for (size_t f = 0; f < m->num_faces; f++) {
            ufbx_face fc = m->faces.data[f];
            if (fc.num_indices < 3) continue;
            if (!insert_unique_face(tri_set, cap - 1, &m->vertex_indices.data[fc.index_begin], fc.num_indices)) {
                continue;
            }
            kept_face_indices[kept_polys++] = (int32_t)f;
            kept_loops += (int32_t)fc.num_indices;
        }
        free(tri_set);

        md->num_polys = kept_polys;
        md->num_loops = kept_loops;
        md->loop_start = (int32_t *)malloc((size_t)kept_polys * sizeof(int32_t));
        md->loop_total = (int32_t *)malloc((size_t)kept_polys * sizeof(int32_t));
        md->poly_mat_index = (int32_t *)calloc((size_t)kept_polys, sizeof(int32_t));
        md->loop_verts = (int32_t *)malloc((size_t)kept_loops * sizeof(int32_t));
        md->loop_normals = (float *)malloc((size_t)kept_loops * 3 * sizeof(float));
        int32_t *kept_orig_corners = (int32_t *)malloc((size_t)kept_loops * sizeof(int32_t));

        ufbx_matrix norm_mat = ufbx_matrix_for_normals(&nd->geometry_to_node);
        int32_t cur_loop = 0;
        for (int32_t p = 0; p < kept_polys; p++) {
            int32_t orig_f = kept_face_indices[p];
            ufbx_face fc = m->faces.data[orig_f];
            md->loop_start[p] = cur_loop;
            md->loop_total[p] = (int32_t)fc.num_indices;
            if (m->face_material.count > (size_t)orig_f) {
                md->poly_mat_index[p] = (int32_t)m->face_material.data[orig_f];
            }
            for (uint32_t k = 0; k < fc.num_indices; k++) {
                uint32_t c_idx = fc.index_begin + k;
                kept_orig_corners[cur_loop] = (int32_t)c_idx;
                md->loop_verts[cur_loop] = (int32_t)m->vertex_indices.data[c_idx];
                ufbx_vec3 n_raw = m->vertex_normal.exists ? ufbx_get_vertex_vec3(&m->vertex_normal, c_idx) : mk_vec3(0, 0, 1);
                ufbx_vec3 n_t = vec3_norm_safe(ufbx_transform_direction(&norm_mat, n_raw), mk_vec3(0, 0, 1));
                md->loop_normals[cur_loop * 3 + 0] = (float)n_t.x;
                md->loop_normals[cur_loop * 3 + 1] = (float)n_t.y;
                md->loop_normals[cur_loop * 3 + 2] = (float)n_t.z;
                cur_loop++;
            }
        }
        free(kept_face_indices);

        /* UVs */
        md->num_uv_layers = (int32_t)m->uv_sets.count;
        if (md->num_uv_layers > 0) {
            md->uv_layers = (ScspUvLayer *)calloc((size_t)md->num_uv_layers, sizeof(ScspUvLayer));
            for (int32_t u = 0; u < md->num_uv_layers; u++) {
                const ufbx_uv_set *uset = &m->uv_sets.data[u];
                md->uv_layers[u].name = uset->name.data;
                md->uv_layers[u].loop_uvs = (float *)malloc((size_t)kept_loops * 2 * sizeof(float));
                for (int32_t l = 0; l < kept_loops; l++) {
                    ufbx_vec2 uv = ufbx_get_vertex_vec2(&uset->vertex_uv, (size_t)kept_orig_corners[l]);
                    md->uv_layers[u].loop_uvs[l * 2 + 0] = (float)uv.x;
                    md->uv_layers[u].loop_uvs[l * 2 + 1] = (float)uv.y;
                }
            }
        }
        /* Append extra AssetStudio LayerElementNormalMapUV DOM UV channels */
        extract_dom_uv_layer(m->element.dom_node, m, kept_orig_corners, kept_loops, &md->uv_layers, &md->num_uv_layers);
        free(kept_orig_corners);

        /* Mesh material indices mapped to scene material table */
        md->num_mats = (int32_t)m->materials.count;
        if (md->num_mats > 0) {
            md->scene_mat_indices = (int32_t *)malloc((size_t)md->num_mats * sizeof(int32_t));
            for (int32_t mm = 0; mm < md->num_mats; mm++) {
                md->scene_mat_indices[mm] = (int32_t)m->materials.data[mm]->typed_id;
            }
        }

        /* Skin weights */
        if (m->skin_deformers.count > 0) {
            ufbx_skin_deformer *sd = m->skin_deformers.data[0];
            size_t total_w = 0;
            for (size_t c = 0; c < sd->clusters.count; c++) {
                ufbx_skin_cluster *cl = sd->clusters.data[c];
                if (!cl->bone_node) continue;
                int32_t b_ni = ctx->ufbx_to_scsp_node[cl->bone_node->typed_id];
                if (b_ni >= 0 && ctx->nodes[b_ni].bone_idx >= 0) {
                    total_w += cl->num_weights;
                }
            }
            md->num_skin_weights = (int32_t)total_w;
            if (total_w > 0) {
                md->skin_bone_idx = (int32_t *)malloc(total_w * sizeof(int32_t));
                md->skin_vert_idx = (int32_t *)malloc(total_w * sizeof(int32_t));
                md->skin_weights = (float *)malloc(total_w * sizeof(float));
                size_t wi = 0;
                for (size_t c = 0; c < sd->clusters.count; c++) {
                    ufbx_skin_cluster *cl = sd->clusters.data[c];
                    if (!cl->bone_node) continue;
                    int32_t b_ni = ctx->ufbx_to_scsp_node[cl->bone_node->typed_id];
                    int32_t bi = (b_ni >= 0) ? ctx->nodes[b_ni].bone_idx : -1;
                    if (bi < 0) continue;
                    for (size_t k = 0; k < cl->num_weights; k++) {
                        md->skin_bone_idx[wi] = bi;
                        md->skin_vert_idx[wi] = (int32_t)cl->vertices.data[k];
                        md->skin_weights[wi] = (float)cl->weights.data[k];
                        wi++;
                    }
                }
            }
        }

        /* Blend shapes (morph targets) */
        if (m->blend_deformers.count > 0) {
            ufbx_blend_deformer *bd = m->blend_deformers.data[0];
            md->num_shapes = (int32_t)bd->channels.count;
            md->shapes = (ScspShapeData *)calloc((size_t)md->num_shapes, sizeof(ScspShapeData));
            for (int32_t s = 0; s < md->num_shapes; s++) {
                ufbx_blend_channel *ch = bd->channels.data[s];
                md->shapes[s].name = ch->name.data;
                ufbx_blend_shape *bs = (ch->keyframes.count > 0)
                    ? ch->keyframes.data[ch->keyframes.count - 1].shape
                    : ch->target_shape;
                if (!bs) continue;
                md->shapes[s].num_offsets = (int32_t)bs->num_offsets;
                md->shapes[s].vert_indices = (int32_t *)malloc(bs->num_offsets * sizeof(int32_t));
                md->shapes[s].deltas_xyz = (float *)malloc(bs->num_offsets * 3 * sizeof(float));
                for (size_t k = 0; k < bs->num_offsets; k++) {
                    md->shapes[s].vert_indices[k] = (int32_t)bs->offset_vertices.data[k];
                    ufbx_vec3 d = ufbx_transform_direction(&nd->geometry_to_node, bs->position_offsets.data[k]);
                    md->shapes[s].deltas_xyz[k * 3 + 0] = (float)d.x;
                    md->shapes[s].deltas_xyz[k * 3 + 1] = (float)d.y;
                    md->shapes[s].deltas_xyz[k * 3 + 2] = (float)d.z;
                }
            }
        }
    }

    return ctx;
}

SCSP_API void scsp_fbx_free(void *handle) {
    if (!handle) return;
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    for (int32_t i = 0; i < ctx->num_meshes; i++) {
        ScspMeshData *md = &ctx->meshes[i];
        free(md->verts_xyz);
        free(md->loop_start);
        free(md->loop_total);
        free(md->poly_mat_index);
        free(md->loop_verts);
        free(md->loop_normals);
        for (int32_t u = 0; u < md->num_uv_layers; u++) {
            free(md->uv_layers[u].loop_uvs);
        }
        free(md->uv_layers);
        free(md->scene_mat_indices);
        free(md->skin_bone_idx);
        free(md->skin_vert_idx);
        free(md->skin_weights);
        for (int32_t s = 0; s < md->num_shapes; s++) {
            free(md->shapes[s].vert_indices);
            free(md->shapes[s].deltas_xyz);
        }
        free(md->shapes);
    }
    free(ctx->meshes);
    free(ctx->bones);
    free(ctx->nodes);
    free(ctx->ufbx_to_scsp_node);
    free(ctx->mats);
    if (ctx->scene) ufbx_free_scene(ctx->scene);
    free(ctx);
}

SCSP_API const char *scsp_fbx_get_armature_name(void *handle) {
    return ((ScspFbxScene *)handle)->armature_name;
}

SCSP_API void scsp_fbx_get_counts(
    void *handle,
    int32_t *out_nodes,
    int32_t *out_bones,
    int32_t *out_meshes,
    int32_t *out_materials
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    *out_nodes = ctx->num_nodes;
    *out_bones = ctx->num_bones;
    *out_meshes = ctx->num_meshes;
    *out_materials = ctx->num_mats;
}

SCSP_API const char *scsp_fbx_get_node_info(
    void *handle,
    int32_t idx,
    int32_t *kind,
    int32_t *parent_idx,
    int32_t *mesh_idx,
    int32_t *bone_idx,
    float mat4_out[16]
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspNodeInfo *n = &ctx->nodes[idx];
    *kind = n->kind;
    *parent_idx = n->parent_idx;
    *mesh_idx = n->mesh_idx;
    *bone_idx = n->bone_idx;
    memcpy(mat4_out, n->world_mat, 16 * sizeof(float));
    return n->name;
}

SCSP_API const char *scsp_fbx_get_bone_info(
    void *handle,
    int32_t bi,
    int32_t *parent_bone_idx,
    float mat4_out[16],
    float *tail_len
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspBoneData *b = &ctx->bones[bi];
    *parent_bone_idx = b->parent_bone_idx;
    *tail_len = b->tail_len;
    memcpy(mat4_out, b->bind_world, 16 * sizeof(float));
    return b->name;
}

SCSP_API const char *scsp_fbx_get_material_info(
    void *handle,
    int32_t mi,
    float diffuse_rgba[4],
    const char **tex_filename,
    const char **tex_rel_filename
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMatInfo *m = &ctx->mats[mi];
    memcpy(diffuse_rgba, m->diffuse_rgba, 4 * sizeof(float));
    *tex_filename = m->tex_filename ? m->tex_filename : "";
    *tex_rel_filename = m->tex_rel_filename ? m->tex_rel_filename : "";
    return m->name;
}

SCSP_API const char *scsp_fbx_get_mesh_header(
    void *handle,
    int32_t mi,
    int32_t *node_index,
    int32_t *num_verts,
    int32_t *num_polys,
    int32_t *num_loops,
    int32_t *num_uv_layers,
    int32_t *num_mats,
    int32_t *num_skin_weights,
    int32_t *num_shapes,
    int32_t *is_skinned,
    int32_t *has_normals
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMeshData *md = &ctx->meshes[mi];
    *node_index = md->node_index;
    *num_verts = md->num_verts;
    *num_polys = md->num_polys;
    *num_loops = md->num_loops;
    *num_uv_layers = md->num_uv_layers;
    *num_mats = md->num_mats;
    *num_skin_weights = md->num_skin_weights;
    *num_shapes = md->num_shapes;
    *is_skinned = md->is_skinned;
    *has_normals = md->has_normals;
    return md->name;
}

SCSP_API void scsp_fbx_fill_mesh_geometry(
    void *handle,
    int32_t mi,
    float *out_verts_xyz,
    int32_t *out_loop_start,
    int32_t *out_loop_total,
    int32_t *out_poly_mat_idx,
    int32_t *out_loop_verts,
    float *out_loop_normals,
    int32_t *out_scene_mats
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMeshData *md = &ctx->meshes[mi];
    if (out_verts_xyz && md->num_verts > 0)
        memcpy(out_verts_xyz, md->verts_xyz, (size_t)md->num_verts * 3 * sizeof(float));
    if (out_loop_start && md->num_polys > 0)
        memcpy(out_loop_start, md->loop_start, (size_t)md->num_polys * sizeof(int32_t));
    if (out_loop_total && md->num_polys > 0)
        memcpy(out_loop_total, md->loop_total, (size_t)md->num_polys * sizeof(int32_t));
    if (out_poly_mat_idx && md->num_polys > 0)
        memcpy(out_poly_mat_idx, md->poly_mat_index, (size_t)md->num_polys * sizeof(int32_t));
    if (out_loop_verts && md->num_loops > 0)
        memcpy(out_loop_verts, md->loop_verts, (size_t)md->num_loops * sizeof(int32_t));
    if (out_loop_normals && md->num_loops > 0)
        memcpy(out_loop_normals, md->loop_normals, (size_t)md->num_loops * 3 * sizeof(float));
    if (out_scene_mats && md->num_mats > 0)
        memcpy(out_scene_mats, md->scene_mat_indices, (size_t)md->num_mats * sizeof(int32_t));
}

SCSP_API const char *scsp_fbx_fill_mesh_uv(void *handle, int32_t mi, int32_t uv_idx, float *out_uv_2f) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMeshData *md = &ctx->meshes[mi];
    if (uv_idx < 0 || uv_idx >= md->num_uv_layers) return "";
    if (out_uv_2f && md->num_loops > 0) {
        memcpy(out_uv_2f, md->uv_layers[uv_idx].loop_uvs, (size_t)md->num_loops * 2 * sizeof(float));
    }
    return md->uv_layers[uv_idx].name;
}

SCSP_API void scsp_fbx_fill_mesh_skin(
    void *handle,
    int32_t mi,
    int32_t *out_bone_idx,
    int32_t *out_vert_idx,
    float *out_weight
) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMeshData *md = &ctx->meshes[mi];
    if (md->num_skin_weights <= 0) return;
    memcpy(out_bone_idx, md->skin_bone_idx, (size_t)md->num_skin_weights * sizeof(int32_t));
    memcpy(out_vert_idx, md->skin_vert_idx, (size_t)md->num_skin_weights * sizeof(int32_t));
    memcpy(out_weight, md->skin_weights, (size_t)md->num_skin_weights * sizeof(float));
}

SCSP_API const char *scsp_fbx_get_shape_header(void *handle, int32_t mi, int32_t si, int32_t *num_offsets) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMeshData *md = &ctx->meshes[mi];
    const ScspShapeData *sd = &md->shapes[si];
    *num_offsets = sd->num_offsets;
    return sd->name;
}

SCSP_API void scsp_fbx_fill_shape_offsets(void *handle, int32_t mi, int32_t si, int32_t *out_vert_idx, float *out_deltas_xyz) {
    ScspFbxScene *ctx = (ScspFbxScene *)handle;
    const ScspMeshData *md = &ctx->meshes[mi];
    const ScspShapeData *sd = &md->shapes[si];
    if (sd->num_offsets <= 0) return;
    memcpy(out_vert_idx, sd->vert_indices, (size_t)sd->num_offsets * sizeof(int32_t));
    memcpy(out_deltas_xyz, sd->deltas_xyz, (size_t)sd->num_offsets * 3 * sizeof(float));
}
