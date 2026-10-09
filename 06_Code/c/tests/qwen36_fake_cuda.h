/* qwen36_fake_cuda.h -- fake CUDA backend shared by the qwen36 tier tests.
 *
 * Defines every coli_cuda_* symbol qwen36_tier.c links against (its
 * signatures come from backend_cuda.h, which the tier includes on its own)
 * and RECORDS what it receives, so a test can assert on real upload/issue
 * traffic without a GPU or the CUDA toolkit. A test that only checked
 * "qt_init returns 1" would pass even with the tier fully broken.
 *
 * Three settable hooks beyond plain recording:
 *   fake_ndev        - device count returned by coli_cuda_available_device_count
 *                       and coli_cuda_device_count (default 1).
 *   fake_issue_hook   - called by coli_cuda_expert_group_issue with the issuing
 *                       device (taken from g[0]->device), the row count and the
 *                       input pointer; its return value is what issue returns.
 *                       NULL (the default) reproduces the old always-0 stub.
 *   fake_upload_hook  - called at the start of every tensor upload, on the
 *                       uploader thread, with the tensor's fmt. A test that
 *                       needs an upload to take TIME (a real cudaMemcpy does)
 *                       sleeps here; NULL (the default) uploads instantly. */
#ifndef QWEN36_FAKE_CUDA_H
#define QWEN36_FAKE_CUDA_H
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../backend_cuda.h"
#include "../gq.h"      /* fake_block_compute: the scalar reference the device kernels must match */

struct ColiCudaTensor { int fmt, I, O, device, gs; const void *w; const float *sc; int owned; };

static int fake_uploads;
static int last_fmt = -1;
static size_t last_bytes;
static const float *last_sc;                  /* scale pointer of the last upload (NULL for a block tensor) */
static unsigned char captured[4096];
static size_t captured_len;

/* sylph (cuda_tier_kquant.md): fake_block_compute=1 makes the fake COMPUTE the
 * raw ggml block formats (fmt 16 + type) with gq_dot_row_ref, so the whole
 * engine can run on the fake tier and its dumps be compared byte for byte
 * with the CPU path. Uploads of block tensors keep a copy of the bytes (the
 * tier frees its staging buffer right after the upload). */
static int fake_block_compute;
static int fake_block_uploads[4];             /* fmt 24 / 28 / 29 / 30 */
static int fake_issues, fake_issue_rows;
static float *fake_group_out; static size_t fake_group_cap;   /* [rows][D] of the last group */
static int fake_block_index(int fmt) { return fmt == 24 ? 0 : fmt == 28 ? 1 : fmt == 29 ? 2 : fmt == 30 ? 3 : -1; }

static int fake_ndev = 1;
static size_t fake_free_bytes = 2ull << 30;    /* what coli_cuda_mem_info reports as free */
static int (*fake_issue_hook)(int device, int count, const float *x) = NULL;
static void (*fake_upload_hook)(int fmt) = NULL;

/* fake_dense_compute=1: coli_cuda_matmul really computes fmt 1 (int8 per
 * row) from the uploaded bytes, so an engine test can put a trunk on the fake
 * tier and demand the same tokens as the CPU int8 reference. The engine
 * frees its int8 rows right after qt_dense_init, so the upload keeps a copy. */
static int fake_dense_compute;
static int upload_common(ColiCudaTensor **t, const void *w, const float *sc, int fmt,
                         int I, int O, int device, int gs) {
    if (fake_upload_hook) fake_upload_hook(fmt);
    const int blk = coli_cuda_block_fmt_supported(fmt);
    /* the real backend's refusals (cuda_tier_kquant.md B3): a block tensor
     * carries no scale array and I is a whole number of blocks */
    if (blk && (sc || !w || I % coli_cuda_block_fmt_elems(fmt))) return 0;
    if (*t) { ColiCudaTensor *c = *t; return c->fmt == fmt && c->I == I && c->O == O && c->device == device; }
    ColiCudaTensor *n = (ColiCudaTensor *)calloc(1, sizeof *n);
    n->fmt = fmt; n->I = I; n->O = O; n->device = device; n->gs = gs; n->w = w; n->sc = sc;
    if (fake_dense_compute && fmt == 1 && w && sc) {
        int8_t *q = (int8_t *)malloc((size_t)I * O); float *s = (float *)malloc((size_t)O * sizeof(float));
        if (q && s) { memcpy(q, w, (size_t)I * O); memcpy(s, sc, (size_t)O * sizeof(float)); n->w = q; n->sc = s; n->owned = 1; }
        else { free(q); free(s); n->w = NULL; n->sc = NULL; }
    }
    size_t blk_bytes = blk ? (size_t)(I / coli_cuda_block_fmt_elems(fmt)) * (size_t)coli_cuda_block_fmt_bytes(fmt) * (size_t)O : 0;
    if (fake_block_compute && blk) {
        uint8_t *c = (uint8_t *)malloc(blk_bytes);
        if (c) { memcpy(c, w, blk_bytes); n->w = c; n->owned = 1; } else n->w = NULL;
    }
    if (blk) fake_block_uploads[fake_block_index(fmt)]++;
    *t = n;
    fake_uploads++;
    last_fmt = fmt; last_sc = sc;
    last_bytes = blk ? blk_bytes : (size_t)I * O / ((fmt == 1 || fmt == 8) ? 1 : 2);
    if (fake_uploads == 1) {
        captured_len = last_bytes < sizeof captured ? last_bytes : sizeof captured;
        memcpy(captured, w, captured_len);
    }
    return 1;
}
int coli_cuda_tensor_upload(ColiCudaTensor **t, const void *w, const float *s,
                            int fmt, int I, int O, int device) {
    return upload_common(t, w, s, fmt, I, O, device, 0);
}
int coli_cuda_tensor_upload_g(ColiCudaTensor **t, const void *w, const float *s,
                              int fmt, int I, int O, int device, int gs) {
    return upload_common(t, w, s, fmt, I, O, device, gs);
}
void coli_cuda_tensor_free(ColiCudaTensor *t) {
    if (t && t->owned) { free((void *)t->w); free((void *)t->sc); }
    free(t);
}
int coli_cuda_available_device_count(void) { return fake_ndev; }
int coli_cuda_device_count(void) { return fake_ndev; }
int coli_cuda_init(const int *d, int n) { (void)d; (void)n; return 1; }
static int fake_lut_published;
int coli_cuda_fp8_set_lut(const float *lut) { fake_lut_published = lut != NULL; return lut != NULL; }
void coli_cuda_shutdown(void) {}
int coli_cuda_mem_info(int device, size_t *freeb, size_t *total) {
    (void)device;
    *freeb = fake_free_bytes; *total = 4ull << 30;   /* 2 GiB liberi by default */
    return 1;
}
/* block-format expert on the host: gate/up GEMVs, h = silu(g).u, down GEMV --
 * gq_dot_row_ref per row (what the device kernels reproduce), host expf */
static int fake_block_expert(const ColiCudaTensor *g, const ColiCudaTensor *u, const ColiCudaTensor *d,
                             const float *x, float *y) {
    if (!g->w || !u->w || !d->w) return 0;
    int D = g->I, Ih = g->O, tg = g->fmt - COLI_FMT_GGML_BASE, tu = u->fmt - COLI_FMT_GGML_BASE, td = d->fmt - COLI_FMT_GGML_BASE;
    if (u->I != D || u->O != Ih || d->I != Ih || d->O != D) return 0;
    float *gg = (float *)malloc((size_t)Ih * sizeof(float)), *uu = (float *)malloc((size_t)Ih * sizeof(float));
    if (!gg || !uu) { free(gg); free(uu); return 0; }
    size_t rg = gq_row_bytes(tg, D), ru = gq_row_bytes(tu, D), rd = gq_row_bytes(td, Ih);
    for (int i = 0; i < Ih; i++) {
        gg[i] = gq_dot_row_ref(tg, (const uint8_t *)g->w + rg * (size_t)i, x, D);
        uu[i] = gq_dot_row_ref(tu, (const uint8_t *)u->w + ru * (size_t)i, x, D);
        float gv = gg[i]; gg[i] = (gv / (1.f + expf(-gv))) * uu[i];
    }
    for (int o = 0; o < D; o++) y[o] = gq_dot_row_ref(td, (const uint8_t *)d->w + rd * (size_t)o, gg, Ih);
    free(gg); free(uu);
    return 1;
}
int coli_cuda_expert_group_issue(ColiCudaTensor *const *g, ColiCudaTensor *const *u,
                                 ColiCudaTensor *const *d, const int *rows,
                                 int count, const float *x) {
    if (fake_issue_hook) return fake_issue_hook(count > 0 ? g[0]->device : -1, count, x);
    if (fake_block_compute && count > 0 && coli_cuda_block_fmt_supported(g[0]->fmt)) {
        int D = g[0]->I, total = 0;
        for (int c = 0; c < count; c++) total += rows[c];
        size_t need = (size_t)total * D * sizeof(float);
        if (need > fake_group_cap) { free(fake_group_out); fake_group_out = (float *)malloc(need); fake_group_cap = fake_group_out ? need : 0; }
        if (!fake_group_out) return 0;
        int off = 0;
        for (int c = 0; c < count; c++) {
            for (int r = 0; r < rows[c]; r++, off++)
                if (!fake_block_expert(g[c], u[c], d[c], x + (size_t)off * D, fake_group_out + (size_t)off * D)) return 0;
        }
        fake_issues++; fake_issue_rows += total;
        return 1;
    }
    (void)u; (void)d; (void)rows;
    return 0;
}
const float *coli_cuda_expert_group_take(int device) { (void)device; return fake_block_compute ? fake_group_out : NULL; }
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d, double *kernel, double *d2h) {
    if (calls) *calls = 0; if (experts) *experts = 0; if (rows) *rows = 0;
    if (h2d) *h2d = 0; if (kernel) *kernel = 0; if (d2h) *d2h = 0;
}
void coli_cuda_stats(int device, size_t *count, size_t *bytes) {
    (void)device; if (count) *count = 0; if (bytes) *bytes = 0;
}

/* dense GEMV on a resident tensor (lm_head / DeltaNet projections placed on a
 * device). Counted, never computed: the placement tests check WHERE work
 * went; the arithmetic has its own oracle in the CUDA build. Parameters are
 * unused on purpose (CFLAGS carry -Wno-unused-parameter). */
static int fake_matmuls, fake_matmul_fail, fake_matmul_rows, fake_matmul_fail_at;
int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device, int gs) {
    fake_matmuls++;
    fake_matmul_rows = S;
    if (fake_matmul_fail || fake_matmuls == fake_matmul_fail_at) {
        for(int i=0;i<S*O;i++) y[i]=NAN;
        return 0;
    }
    ColiCudaTensor *t = tensor ? *tensor : NULL;
    if (fake_dense_compute && t && t->fmt == 1 && t->w && t->sc && t->I == I && t->O == O) {
        const int8_t *q = (const int8_t *)t->w;
        for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
            const int8_t *w = q + (size_t)o * I; const float *xs = x + (size_t)s * I; float a = 0.f;
            for (int i = 0; i < I; i++) a += xs[i] * (float)w[i];
            y[(size_t)s * O + o] = a * t->sc[o];
        }
    }
    if (fake_block_compute && t && coli_cuda_block_fmt_supported(t->fmt) && t->w && t->I == I && t->O == O) {
        int type = t->fmt - COLI_FMT_GGML_BASE; size_t rb = gq_row_bytes(type, I);
        for (int s = 0; s < S; s++) for (int o = 0; o < O; o++)
            y[(size_t)s * O + o] = gq_dot_row_ref(type, (const uint8_t *)t->w + rb * (size_t)o, x + (size_t)s * I, I);
    }
    return 1;
}

#endif /* QWEN36_FAKE_CUDA_H */
