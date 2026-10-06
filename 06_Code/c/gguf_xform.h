/* gguf_xform.h — undo, losslessly, what llama.cpp's converter did to the
 * Qwen3.5/3.6 MoE tensors (architecture §7, audited against the owner's
 * HF-named container on 2026-10-05):
 *
 *   RMSNorm weights stored as 1 + w          → w = stored − 1
 *   ssm_a stored as −exp(A_log)              → A_log = log(−a)        (the one non-bit-exact step: 2 ulp)
 *   DeltaNet value heads reordered           → gather back to HF order
 *
 * The permutation: GGUF head j holds HF head h(j) = (j mod vk)·(vh/vk) + j div vk
 * (llama.cpp's grouped recurrence wants v-head i to read k-head i mod vk; the
 * engine's deltanet() reads k-head i / (vh/vk) and stays untouched). For
 * vh = 32, vk = 16 this is [0,2,…,30,1,3,…,31]. Undoing it moves whole head
 * blocks: rows (attn_gate, ssm_alpha/beta, the value third of attn_qkv and of
 * ssm_conv1d), elements (ssm_dt.bias, ssm_a) or input columns (ssm_out). Quantized
 * rows move as raw block rows; permuted columns inside quantized rows move as
 * whole blocks when the head width is a multiple of the block size (vdim 128 vs
 * 32 for Q8_0 on the real model), otherwise the caller dequantizes first.
 *
 * Pure functions on caller buffers, dst != src. Header-only, static. */
#ifndef COLI_GGUF_XFORM_H
#define COLI_GGUF_XFORM_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* HF value head stored at GGUF head j */
static inline int gx_hf_head(int j, int vh, int vk) { int r = vh / vk; return (j % vk) * r + j / vk; }
/* GGUF head holding HF value head h (the inverse) */
static inline int gx_gguf_head(int h, int vh, int vk) { int r = vh / vk; return (h % r) * vk + h / r; }
static inline int gx_perm_ok(int vh, int vk) { return vh > 0 && vk > 0 && vh % vk == 0; }

/* stored 1+w → w, in place */
static inline void gx_norm_unplus1(float *w, int64_t n) { for (int64_t i = 0; i < n; i++) w[i] = w[i] - 1.0f; }

/* stored a = −exp(A_log) → A_log = log(−a), in place; returns the number of
 * non-negative entries (which have no logarithm: a broken file) */
static inline int gx_alog_from_a(float *a, int64_t n) {
    int bad = 0;
    for (int64_t i = 0; i < n; i++) { if (!(a[i] < 0.f)) { bad++; a[i] = 0.f; } else a[i] = logf(-a[i]); }
    return bad;
}

/* Rows [off, off + vh·per) of a row-major matrix (row = row_bytes) are vh
 * blocks of `per` rows in GGUF head order; copy the whole matrix to dst with
 * those blocks back in HF order. nrows is the full row count. */
static inline void gx_unperm_rowblocks(uint8_t *dst, const uint8_t *src, int64_t nrows, size_t row_bytes,
                                       int64_t off, int vh, int vk, int per) {
    if (off) memcpy(dst, src, (size_t)off * row_bytes);
    for (int h = 0; h < vh; h++) {
        int j = gx_gguf_head(h, vh, vk);
        memcpy(dst + ((size_t)off + (size_t)h * per) * row_bytes, src + ((size_t)off + (size_t)j * per) * row_bytes, (size_t)per * row_bytes);
    }
    int64_t tail = off + (int64_t)vh * per;
    if (tail < nrows) memcpy(dst + (size_t)tail * row_bytes, src + (size_t)tail * row_bytes, (size_t)(nrows - tail) * row_bytes);
}
static inline void gx_unperm_rowblocks_f32(float *dst, const float *src, int64_t nrows, int64_t cols, int64_t off, int vh, int vk, int per) {
    gx_unperm_rowblocks((uint8_t *)dst, (const uint8_t *)src, nrows, (size_t)cols * sizeof(float), off, vh, vk, per);
}
/* one element per value head (dt_bias, ssm_a) */
static inline void gx_unperm_elems_f32(float *dst, const float *src, int vh, int vk) {
    for (int h = 0; h < vh; h++) dst[h] = src[gx_gguf_head(h, vh, vk)];
}

/* Columns of an [nrows][vh·per] f32 matrix are vh blocks of `per` in GGUF head
 * order; copy to dst with the blocks back in HF order. */
static inline void gx_unperm_colblocks_f32(float *dst, const float *src, int64_t nrows, int vh, int vk, int per) {
    int64_t width = (int64_t)vh * per;
    for (int64_t r = 0; r < nrows; r++)
        for (int h = 0; h < vh; h++)
            memcpy(dst + r * width + (int64_t)h * per, src + r * width + (int64_t)gx_gguf_head(h, vh, vk) * per, (size_t)per * sizeof(float));
}
/* Same on quantized rows: a column block of `per` elements must be a whole
 * number of quant blocks (per % block_elems == 0). Returns 0, -1 if it is not. */
static inline int gx_unperm_colblocks_raw(uint8_t *dst, const uint8_t *src, int64_t nrows, int vh, int vk, int per,
                                          int block_elems, size_t block_bytes) {
    if (per % block_elems) return -1;
    size_t bph = (size_t)(per / block_elems) * block_bytes;      /* bytes per head block */
    size_t row_bytes = (size_t)vh * bph;
    for (int64_t r = 0; r < nrows; r++)
        for (int h = 0; h < vh; h++)
            memcpy(dst + (size_t)r * row_bytes + (size_t)h * bph, src + (size_t)r * row_bytes + (size_t)gx_gguf_head(h, vh, vk) * bph, bph);
    return 0;
}

#endif /* COLI_GGUF_XFORM_H */
