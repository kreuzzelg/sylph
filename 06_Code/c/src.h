/* src.h — tensor-source façade: one HF-named request, two containers.
 *
 * The engine (qwen36.c) loads weights by the HF names its converter keeps
 * (`model.layers.<i>.<kind>`, `model.embed_tokens.weight`, ...). This header
 * answers those requests from either
 *   - the safetensors container, by delegating VERBATIM to st.h (the arm the
 *     oracles already pin: nothing changes there, NFR-5), or
 *   - a GGUF written by llama.cpp's converter (gguf.h, phase 1), translating
 *     the name with qwen35_names.h, decoding the blocks exactly with gq.h
 *     (phase 2) and undoing the converter's value transforms with
 *     gguf_xform.h — so the engine's math never sees a GGUF-specific value.
 *
 * Dense GGUF matrices can also be handed over as stored: Q8_0 as an int8
 * plane + f32 scale per 32 (gsgemv.h's layout, lossless), K-quants as raw
 * block rows for gq_matmul. Routed experts are three slices of the 3-D expert
 * tensors (ts_expert), read with pread into the caller's slot.
 *
 * Errors follow the loaders' discipline: a missing or unsupported tensor
 * names the tensor, both spellings and the rule, then exits. Header-only,
 * static; the GGUF geometry needed for the un-permutations (value heads) is
 * set by the caller once the config is known. */
#ifndef COLI_SRC_H
#define COLI_SRC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>
#include "compat.h"
#include "st.h"
#include "gguf.h"
#include "gq.h"
#include "gguf_xform.h"
#include "qwen35_names.h"

typedef struct {
    int gguf;                 /* 0 = safetensors container (st.h), 1 = GGUF */
    shards *S;                /* container arm: the engine's shards, owned by the caller */
    GgufSet G;                /* GGUF arm */
    char path[2048];          /* as given */
    char dir[2048], stem[256];/* GGUF: directory and stem (sidecars live in <dir>/.coli-<stem>/) */
    /* geometry for the value-head un-permutations (GGUF only; from ts_cfg) */
    int vh, vk, vdim, kdim, convk;
    int n_layers, nextn;      /* trunk blocks and skipped NextN blocks */
    int n_attn;               /* attention blocks (set by cfg_from_gguf; for the startup line) */
    /* FR-30 statistics: every expert slice read goes through ts_read_expert_slice */
    uint64_t rd_slices, rd_bytes;
    uint8_t  touched[GGUF_MAX_SPLITS];   /* parts that served at least one slice */
} TensorSource;

/* FR-29: where sidecars of this model would live. GGUF: <dir>/.coli-<stem>/ (never beside
 * the .gguf); container: the model directory itself, as every engine does today. The
 * directory is not created here -- only whoever writes a sidecar creates it. */
static void ts_sidecar_dir(const TensorSource *ts, char *out, size_t cap) {
    if (ts->gguf) snprintf(out, cap, "%.1800s/.coli-%.250s/", ts->dir, ts->stem);
    else snprintf(out, cap, "%.2000s", ts->path);
}

/* A .gguf file (single or split part), or a directory that holds *.gguf and no
 * config.json (a container directory always has one). */
static int ts_is_gguf_path(const char *path) {
    struct stat sb;
    if (!path || stat(path, &sb)) return 0;
    if (S_ISDIR(sb.st_mode)) {
        char p[2304]; snprintf(p, sizeof p, "%s/config.json", path);
        if (!stat(p, &sb)) return 0;
        DIR *d = opendir(path); if (!d) return 0;
        struct dirent *e; int found = 0;
        while (!found && (e = readdir(d))) { size_t n = strlen(e->d_name); found = n > 5 && !strcasecmp(e->d_name + n - 5, ".gguf"); }
        closedir(d); return found;
    }
    size_t n = strlen(path);
    return n > 5 && !strcasecmp(path + n - 5, ".gguf");
}

static void ts_split_path(TensorSource *ts) {
    struct stat sb;
    if (!stat(ts->path, &sb) && S_ISDIR(sb.st_mode)) {
        snprintf(ts->dir, sizeof ts->dir, "%s", ts->path);
        snprintf(ts->stem, sizeof ts->stem, "%.250s", ts->G.nfiles ? gguf_basename(ts->G.files[0].path) : "model");
    } else {
        const char *b = gguf_basename(ts->path);
        size_t dl = (size_t)(b - ts->path); if (dl && dl < sizeof ts->dir) { memcpy(ts->dir, ts->path, dl - 1); ts->dir[dl - 1] = 0; } else snprintf(ts->dir, sizeof ts->dir, ".");
        snprintf(ts->stem, sizeof ts->stem, "%.250s", b);
    }
    size_t n = strlen(ts->stem);
    if (n > 5 && !strcasecmp(ts->stem + n - 5, ".gguf")) ts->stem[n - 5] = 0;
    n = strlen(ts->stem);                                   /* "-00001-of-00011" */
    if (n > 15 && ts->stem[n - 15] == '-' && !strncmp(ts->stem + n - 9, "-of-", 4)) ts->stem[n - 15] = 0;
}

/* Open the source. Container: st_init_multi(S, snap, extra_dirs), as before.
 * GGUF: gguf_open_set (exits on a malformed file, like st_init does). */
static void ts_init(TensorSource *ts, shards *S, const char *snap, const char *extra_dirs) {
    memset(ts, 0, sizeof *ts);
    snprintf(ts->path, sizeof ts->path, "%s", snap);
    ts->S = S;
    if (!ts_is_gguf_path(snap)) { st_init_multi(S, snap, extra_dirs); return; }
    ts->gguf = 1;
    gguf_open_set_or_die(&ts->G, snap, extra_dirs);
    ts_split_path(ts);
    const char *arch = gguf_kv_str(&ts->G, "general.architecture");
    if (!arch || strcmp(arch, "qwen35moe")) {
        fprintf(stderr, "[GGUF] %s: general.architecture is '%s'; this engine reads qwen35moe (Qwen3.5/3.6 MoE) -- refusing\n", snap, arch ? arch : "(missing)"); exit(1);
    }
}
static void ts_close(TensorSource *ts) { if (ts->gguf) gguf_close(&ts->G); }

/* ---- lookup ---------------------------------------------------------------------------- */
typedef struct {
    st_tensor *st; GgufTensor *gt;
    int64_t numel, nbytes;
    int type;                 /* ggml type (GGUF) or -1 */
    int xform, layer, expert; /* QN_X_*, block, expert slice (-1 unless a per-expert HF name) */
    char gname[256];
} TsTensor;

static int ts_find(TensorSource *ts, const char *hf, TsTensor *t) {
    memset(t, 0, sizeof *t); t->type = -1; t->layer = -1; t->expert = -1;
    if (!ts->gguf) {
        t->st = st_find(ts->S, hf);
        if (!t->st) return -1;
        t->numel = t->st->numel; t->nbytes = t->st->nbytes; return 0;
    }
    t->xform = qn_to_gguf(hf, t->gname, sizeof t->gname, &t->layer, &t->expert);
    if (t->xform < 0) return -1;
    t->gt = gguf_find(&ts->G, t->gname);
    if (!t->gt) return -1;
    t->type = t->gt->type;
    int64_t n = 1; for (int d = 0; d < t->gt->n_dims; d++) n *= t->gt->ne[d];
    if (t->expert >= 0) {               /* one slice of the 3-D expert tensor */
        if (t->gt->n_dims < 3 || t->expert >= t->gt->ne[2]) return -1;
        n = t->gt->ne[0] * t->gt->ne[1];
        t->nbytes = t->gt->ne[1] * gguf_row_size(t->gt->type, t->gt->ne[0]);
    } else t->nbytes = t->gt->nbytes;
    t->numel = n;
    return 0;
}
static int ts_has(TensorSource *ts, const char *hf) { TsTensor t; return ts_find(ts, hf, &t) == 0; }
static int64_t ts_numel(TensorSource *ts, const char *hf) { TsTensor t; return ts_find(ts, hf, &t) ? -1 : t.numel; }
/* ggml type of a GGUF tensor, -1 for the container (or a missing name) */
static int ts_ggml_type(TensorSource *ts, const char *hf) { TsTensor t; return ts_find(ts, hf, &t) ? -1 : t.type; }

static void ts_pread(TensorSource *ts, int file, int64_t off, void *dst, size_t n, const char *what) {
    int fd = ts->G.files[file].fd; size_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, (char *)dst + got, n - got, off + (int64_t)got);
        if (r <= 0) { fprintf(stderr, "[GGUF] short read of %s at %lld (+%zu of %zu): %s\n", what, (long long)off, got, n, r < 0 ? strerror(errno) : "EOF"); exit(1); }
        got += (size_t)r;
    }
}
static void ts_refuse_type(const char *hf, const TsTensor *t) {
    fprintf(stderr, "[GGUF] unsupported type %s for %s (GGUF %s) -- refusing; this build reads F32 F16 BF16 Q4_0 Q8_0 Q4_K Q5_K Q6_K\n",
            gguf_type_name(t->type), hf, t->gname); exit(1);
}
static void ts_refuse_missing(TensorSource *ts, const char *hf) {
    char g[256]; int l, e; int x = ts->gguf ? qn_to_gguf(hf, g, sizeof g, &l, &e) : 0;
    if (ts->gguf && x < 0) fprintf(stderr, "[GGUF] missing %s: the name is not in the qwen35moe table (qwen35_names.h)\n", hf);
    else if (ts->gguf) fprintf(stderr, "[GGUF] missing %s (GGUF %s)\n", hf, g);
    else fprintf(stderr, "missing %s\n", hf);
    exit(1);
}
static int ts_perm_needed(int xform) {
    return xform == QN_X_SSM_A || xform == QN_X_PERM_HEAD_ROWS || xform == QN_X_PERM_HEAD_ELEMS ||
           xform == QN_X_PERM_QKV_ROWS || xform == QN_X_PERM_CONV_ROWS || xform == QN_X_PERM_HEAD_COLS;
}
static void ts_need_perm(TensorSource *ts, const char *hf) {
    if (!gx_perm_ok(ts->vh, ts->vk) || ts->vdim <= 0 || ts->kdim <= 0) {
        fprintf(stderr, "[GGUF] %s needs the DeltaNet head geometry (vh=%d vk=%d vdim=%d kdim=%d) before it can be read\n", hf, ts->vh, ts->vk, ts->vdim, ts->kdim); exit(1);
    }
}

/* ---- f32 read: exact decode + un-transform -------------------------------------------- */
static void ts_read_f32(TensorSource *ts, const char *hf, float *out, int64_t numel) {
    if (!ts->gguf) { st_read_f32(ts->S, hf, out, 0); return; }     /* verbatim delegation */
    TsTensor t;
    if (ts_find(ts, hf, &t)) ts_refuse_missing(ts, hf);
    if (t.expert >= 0) { fprintf(stderr, "[GGUF] %s is an expert slice: use ts_expert\n", hf); exit(1); }
    if (!gq_supported(t.type)) ts_refuse_type(hf, &t);
    if (numel > 0 && t.numel != numel) {
        fprintf(stderr, "%s (GGUF %s): %lld elements, config implies %lld -- refusing\n", hf, t.gname, (long long)t.numel, (long long)numel); exit(1);
    }
    if (t.numel > INT32_MAX) { fprintf(stderr, "[GGUF] %s: %lld elements exceed the loader's range\n", hf, (long long)t.numel); exit(1); }
    uint8_t *raw = malloc((size_t)t.nbytes);
    float *tmp = ts_perm_needed(t.xform) ? malloc((size_t)t.numel * sizeof(float)) : out;
    if (!raw || !tmp) { fprintf(stderr, "OOM reading %s\n", hf); exit(1); }
    ts_pread(ts, t.gt->file, t.gt->off, raw, (size_t)t.nbytes, t.gname);
    if (gq_deq_row(t.type, raw, tmp, (int)t.numel)) { fprintf(stderr, "[GGUF] %s: %lld elements are not a whole number of %s blocks\n", t.gname, (long long)t.numel, gguf_type_name(t.type)); exit(1); }
    free(raw);
    int64_t ne0 = t.gt->ne[0], ne1 = t.gt->n_dims > 1 ? t.gt->ne[1] : 1;
    switch (t.xform) {
    case QN_X_NONE: break;
    case QN_X_NORM_PLUS1: gx_norm_unplus1(out, t.numel); break;
    case QN_X_SSM_A:
        ts_need_perm(ts, hf);
        if (t.numel != ts->vh) { fprintf(stderr, "[GGUF] %s has %lld entries, %d value heads expected\n", t.gname, (long long)t.numel, ts->vh); exit(1); }
        gx_unperm_elems_f32(out, tmp, ts->vh, ts->vk);
        if (gx_alog_from_a(out, t.numel)) { fprintf(stderr, "[GGUF] %s holds non-negative entries; expected -exp(A_log) -- refusing\n", t.gname); exit(1); }
        break;
    case QN_X_PERM_HEAD_ELEMS:
        ts_need_perm(ts, hf);
        if (t.numel != ts->vh) { fprintf(stderr, "[GGUF] %s has %lld entries, %d value heads expected\n", t.gname, (long long)t.numel, ts->vh); exit(1); }
        gx_unperm_elems_f32(out, tmp, ts->vh, ts->vk); break;
    case QN_X_PERM_HEAD_ROWS:
        ts_need_perm(ts, hf);
        if (ne1 % ts->vh) { fprintf(stderr, "[GGUF] %s: %lld rows are not %d value heads\n", t.gname, (long long)ne1, ts->vh); exit(1); }
        gx_unperm_rowblocks_f32(out, tmp, ne1, ne0, 0, ts->vh, ts->vk, (int)(ne1 / ts->vh)); break;
    case QN_X_PERM_QKV_ROWS: case QN_X_PERM_CONV_ROWS: {
        ts_need_perm(ts, hf);
        int64_t off = 2LL * ts->vk * ts->kdim;
        if (ne1 != off + (int64_t)ts->vh * ts->vdim) { fprintf(stderr, "[GGUF] %s: %lld rows != 2*vk*kdim + vh*vdim (%lld)\n", t.gname, (long long)ne1, (long long)(off + (int64_t)ts->vh * ts->vdim)); exit(1); }
        gx_unperm_rowblocks_f32(out, tmp, ne1, ne0, off, ts->vh, ts->vk, ts->vdim); break; }
    case QN_X_PERM_HEAD_COLS:
        ts_need_perm(ts, hf);
        if (ne0 != (int64_t)ts->vh * ts->vdim) { fprintf(stderr, "[GGUF] %s: %lld columns != vh*vdim (%d)\n", t.gname, (long long)ne0, ts->vh * ts->vdim); exit(1); }
        gx_unperm_colblocks_f32(out, tmp, ne1, ts->vh, ts->vk, ts->vdim); break;
    default: break;
    }
    if (tmp != out) free(tmp);
}

/* ---- dense matrices as stored ----------------------------------------------------------- */
/* GGUF Q8_0 [O][I] → int8 plane + f32 scale per 32 (lossless), rows/columns
 * back in HF order at block granularity. 0 on success; -1 when the tensor is
 * not Q8_0 or a column permutation cannot move whole blocks (caller: f32). */
static int ts_read_q8_split(TensorSource *ts, const char *hf, int I, int O, int8_t *plane, float *scales) {
    if (!ts->gguf) return -1;
    TsTensor t;
    if (ts_find(ts, hf, &t)) ts_refuse_missing(ts, hf);
    if (t.type != GQ_Q8_0 || t.expert >= 0) return -1;
    if (t.gt->ne[0] != I || (t.gt->n_dims > 1 ? t.gt->ne[1] : 1) != O) { fprintf(stderr, "[GGUF] %s: shape {%lld,%lld}, config implies {%d,%d} -- refusing\n", t.gname, (long long)t.gt->ne[0], (long long)(t.gt->n_dims > 1 ? t.gt->ne[1] : 1), I, O); exit(1); }
    if (t.xform == QN_X_PERM_HEAD_COLS && (ts->vdim % 32)) return -1;
    if (ts_perm_needed(t.xform)) ts_need_perm(ts, hf);
    size_t rb = gq_row_bytes(GQ_Q8_0, I), n = rb * (size_t)O;
    uint8_t *raw = malloc(n), *fix = ts_perm_needed(t.xform) ? malloc(n) : raw;
    if (!raw || !fix) { fprintf(stderr, "OOM reading %s\n", hf); exit(1); }
    ts_pread(ts, t.gt->file, t.gt->off, raw, n, t.gname);
    switch (t.xform) {
    case QN_X_PERM_HEAD_ROWS: gx_unperm_rowblocks(fix, raw, O, rb, 0, ts->vh, ts->vk, O / ts->vh); break;
    case QN_X_PERM_QKV_ROWS:  gx_unperm_rowblocks(fix, raw, O, rb, 2LL * ts->vk * ts->kdim, ts->vh, ts->vk, ts->vdim); break;
    case QN_X_PERM_HEAD_COLS: gx_unperm_colblocks_raw(fix, raw, O, ts->vh, ts->vk, ts->vdim, 32, 34); break;
    default: break;
    }
    gq_q8_0_split(fix, I, O, plane, scales);
    if (fix != raw) free(fix);
    free(raw);
    return 0;
}
/* Raw block rows of a GGUF dense matrix [O][I] of a supported quantized type
 * (K-quants for gq_matmul). Only tensors without a permutation (lm_head,
 * attention projections, shared experts). *dst is malloc'd. 0 / -1. */
static int ts_read_raw_rows(TensorSource *ts, const char *hf, int I, int O, uint8_t **dst, size_t *nbytes) {
    if (!ts->gguf) return -1;
    TsTensor t;
    if (ts_find(ts, hf, &t)) ts_refuse_missing(ts, hf);
    if (!gq_supported(t.type) || t.type == GQ_F32 || t.type == GQ_F16 || t.type == GQ_BF16 || t.expert >= 0 || ts_perm_needed(t.xform)) return -1;
    if (t.gt->ne[0] != I || (t.gt->n_dims > 1 ? t.gt->ne[1] : 1) != O) { fprintf(stderr, "[GGUF] %s: shape {%lld,%lld}, config implies {%d,%d} -- refusing\n", t.gname, (long long)t.gt->ne[0], (long long)(t.gt->n_dims > 1 ? t.gt->ne[1] : 1), I, O); exit(1); }
    size_t n = gq_row_bytes(t.type, I) * (size_t)O;
    *dst = malloc(n); if (!*dst) { fprintf(stderr, "OOM reading %s\n", hf); exit(1); }
    ts_pread(ts, t.gt->file, t.gt->off, *dst, n, t.gname);
    *nbytes = n; return 0;
}

/* ---- routed experts ------------------------------------------------------------------------ */
typedef struct {
    int type[3];              /* gate, up, down ggml types */
    int file[3]; int64_t off[3]; size_t bytes[3];
    int64_t rows[3], cols[3]; /* gate/up [F][H], down [H][F] */
} TsExpert;
/* Slice `eid` of block `layer`'s three expert tensors. 0, or -1 if the block
 * has no 3-D expert tensors (container, or missing). Exits on an unsupported
 * type or a malformed tensor. */
static int ts_expert(TensorSource *ts, int layer, int eid, TsExpert *e) {
    if (!ts->gguf) return -1;
    static const char *names[3] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
    for (int k = 0; k < 3; k++) {
        char g[128]; snprintf(g, sizeof g, "blk.%d.%s", layer, names[k]);
        GgufTensor *t = gguf_find(&ts->G, g);
        if (!t) { if (k == 0) return -1; fprintf(stderr, "[GGUF] missing %s (block %d has %s but not this one)\n", g, layer, names[0]); exit(1); }
        if (!gq_supported(t->type)) { fprintf(stderr, "[GGUF] unsupported type %s for %s -- refusing; this build reads F32 F16 BF16 Q4_0 Q8_0 Q4_K Q5_K Q6_K\n", gguf_type_name(t->type), g); exit(1); }
        if (t->n_dims != 3 || eid < 0 || eid >= t->ne[2]) { fprintf(stderr, "[GGUF] %s: expert %d of %lld (dims %d) -- refusing\n", g, eid, (long long)(t->n_dims == 3 ? t->ne[2] : 0), t->n_dims); exit(1); }
        int64_t rb = gguf_row_size(t->type, t->ne[0]);
        if (rb <= 0) { fprintf(stderr, "[GGUF] %s: ne0 %lld is not a whole number of %s blocks\n", g, (long long)t->ne[0], gguf_type_name(t->type)); exit(1); }
        e->type[k] = t->type; e->file[k] = t->file; e->cols[k] = t->ne[0]; e->rows[k] = t->ne[1];
        e->bytes[k] = (size_t)(rb * t->ne[1]); e->off[k] = t->off + (int64_t)eid * (int64_t)e->bytes[k];
    }
    return 0;
}
static void ts_read_expert_slice(TensorSource *ts, const TsExpert *e, int which, uint8_t *dst) {
    ts_pread(ts, e->file[which], e->off[which], dst, e->bytes[which], which == 0 ? "ffn_gate_exps slice" : which == 1 ? "ffn_up_exps slice" : "ffn_down_exps slice");
    /* one pread per slice (FR-27); the counters are what the "GGUF reads:" line prints */
    __atomic_add_fetch(&ts->rd_slices, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&ts->rd_bytes, (uint64_t)e->bytes[which], __ATOMIC_RELAXED);
    if (e->file[which] >= 0 && e->file[which] < GGUF_MAX_SPLITS) ts->touched[e->file[which]] = 1;
}
static int ts_parts_touched(const TensorSource *ts) { int n = 0; for (int i = 0; i < ts->G.nfiles && i < GGUF_MAX_SPLITS; i++) n += ts->touched[i] != 0; return n; }

/* Raw rows of a dense 2-D tensor of ANY supported type (F32/F16/BF16 included), no
 * un-transform: for tables the engine decodes row by row (token_embd via gq_embed_row).
 * -1 when the tensor is permuted or an expert slice (the caller dequantizes instead). */
static int ts_read_rows_any(TensorSource *ts, const char *hf, int I, int O, uint8_t **dst, size_t *nbytes, int *type) {
    if (!ts->gguf) return -1;
    TsTensor t;
    if (ts_find(ts, hf, &t)) ts_refuse_missing(ts, hf);
    if (!gq_supported(t.type) || t.expert >= 0 || ts_perm_needed(t.xform) || t.xform == QN_X_NORM_PLUS1) return -1;
    if (t.gt->ne[0] != I || (t.gt->n_dims > 1 ? t.gt->ne[1] : 1) != O) { fprintf(stderr, "[GGUF] %s: shape {%lld,%lld}, config implies {%d,%d} -- refusing\n", t.gname, (long long)t.gt->ne[0], (long long)(t.gt->n_dims > 1 ? t.gt->ne[1] : 1), I, O); exit(1); }
    size_t n = gq_row_bytes(t.type, I) * (size_t)O;
    *dst = malloc(n); if (!*dst) { fprintf(stderr, "OOM reading %s\n", hf); exit(1); }
    ts_pread(ts, t.gt->file, t.gt->off, *dst, n, t.gname);
    *nbytes = n; *type = t.type; return 0;
}
static size_t ts_expert_bytes(const TsExpert *e) { return e->bytes[0] + e->bytes[1] + e->bytes[2]; }

/* ---- the startup line (FR-30, format fixed in 07_Tests/IntegrationTest/expert_streaming.md) --
 *   [GGUF] <arch> · <B> blocks (<A> attention) · <P> part(s) · experts <tg>/<tu>/<td> <x> MB each × <E> × <B> = <y> GB
 *          · dense <type list> <z> GB · token_embd <type> <mode> · output <type> <kernel> · experts on CPU (gq_moe_run)
 *          · sidecars <dir>/.coli-<stem>/
 * `embd_mode` and `out_kernel` are the engine's decisions (it knows its env and kernels);
 * everything else is read off the index. */
static void ts_describe(TensorSource *ts, char *buf, size_t cap, const char *embd_mode, const char *out_kernel, size_t slot_bytes) {
    if (!ts->gguf) { snprintf(buf, cap, "safetensors container %.900s", ts->path); return; }
    int cnt[GGML_TYPE_COUNT] = {0}; int64_t dense = 0, experts = 0, n_exp = 0;
    for (int64_t i = 0; i < ts->G.nt; i++) {
        GgufTensor *t = &ts->G.t[i]; const char *n = gguf_tensor_name(&ts->G, t);
        if (strstr(n, "_exps.")) { experts += t->nbytes > 0 ? t->nbytes : 0; if (t->n_dims == 3 && !n_exp) n_exp = t->ne[2]; }
        else { dense += t->nbytes > 0 ? t->nbytes : 0; if (t->type >= 0 && t->type < GGML_TYPE_COUNT) cnt[t->type]++; }
    }
    TsExpert e; const char *tg = "?", *tu = "?", *td = "?";
    if (ts->n_layers > 0 && ts_expert(ts, 0, 0, &e) == 0) { tg = gguf_type_name(e.type[0]); tu = gguf_type_name(e.type[1]); td = gguf_type_name(e.type[2]); }
    GgufTensor *emb = gguf_find(&ts->G, "token_embd.weight"), *outw = gguf_find(&ts->G, "output.weight");
    const char *arch = gguf_kv_str(&ts->G, "general.architecture");
    char side[2304]; ts_sidecar_dir(ts, side, sizeof side);
    int w = snprintf(buf, cap, "[GGUF] %s · %d blocks (%d attention)%s · %d part%s · experts %s/%s/%s %.2f MB each × %lld × %d = %.2f GB · dense",
                     arch ? arch : "?", ts->n_layers, ts->n_attn, ts->nextn ? " (+NextN skipped)" : "", ts->G.nfiles, ts->G.nfiles == 1 ? "" : "s",
                     tg, tu, td, slot_bytes / 1048576.0, (long long)n_exp, ts->n_layers, experts / 1073741824.0);
    for (int t = 0; t < GGML_TYPE_COUNT && w < (int)cap; t++) if (cnt[t]) w += snprintf(buf + w, cap - (size_t)w, " %s×%d", gguf_type_name(t), cnt[t]);
    if (w < (int)cap) w += snprintf(buf + w, cap - (size_t)w, " %.2f GB · token_embd %s %s · output %s %s · experts on CPU (gq_moe_run) · sidecars %s",
                                    dense / 1073741824.0, emb ? gguf_type_name(emb->type) : "?", embd_mode, outw ? gguf_type_name(outw->type) : "?", out_kernel, side);
}

#endif /* COLI_SRC_H */
