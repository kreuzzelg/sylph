/* gq.h — ggml block formats (GGUF) on the CPU.
 *
 *   table     gq_types[]: the v1 set F32 F16 BF16 Q4_0 Q8_0 Q4_K Q5_K Q6_K with
 *             ggml's ids and block geometry; gq_supported / gq_row_bytes.
 *   dequant   gq_deq_row(): bit-for-bit ggml's dequantize_row_<type> (the E0
 *             contract of 07_Tests/IntegrationTest/gq_kernels.md): same expression
 *             order, no fused multiply-add, so the sign of every zero and the
 *             last bit of every value equal llama.cpp's reference decode.
 *   dot       gq_dot_row(): f32 activations against raw blocks. The scalar
 *             reference defines the numerics (8 lanes, lane = element & 7, fma
 *             per element, the block scale folded per lane, one fixed 8->1 tree
 *             at the end, exactly as expert_ffn.h does for planar int4); the
 *             AVX2 and NEON paths reproduce it bit for bit.
 *   dense     gq_matmul() (lm_head Q6_K / Q8_0, OMP over rows), gq_embed_row().
 *   split     gq_q8_0_split(): Q8_0 -> int8 plane + f32 scale per 32, lossless,
 *             the exact input gsgemv.h's matmul_q_gs(gs=32) expects.
 *   layer     gq_moe_run(): the K-quant twin of xf_moe_run: gate+up fused over
 *             the experts a layer touched, silu, down, K contributions summed
 *             in rank order. Same work split, same scratch discipline, so
 *             moe() dispatches on the slot flavour and nothing else changes.
 *
 * Header-only, all static, no I/O, no allocation beyond the caller's scratch.
 * Block layouts follow the public ggml definitions (ggml-common.h); no ggml
 * code is copied. */
#ifndef COLI_GQ_H
#define COLI_GQ_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define GQ_HAVE_NEON 1
#endif
#if defined(__AVX2__) && defined(__FMA__)
#define GQ_HAVE_AVX2 1
#endif

/* ---- types ------------------------------------------------------------------- */

enum { GQ_F32 = 0, GQ_F16 = 1, GQ_Q4_0 = 2, GQ_Q8_0 = 8, GQ_Q4_K = 12, GQ_Q5_K = 13,
       GQ_Q6_K = 14, GQ_BF16 = 30, GQ_TYPE_COUNT = 40 };
#define GQ_QK_K 256

typedef struct { const char *name; int block; int tsize; } GqTypeInfo;
static const GqTypeInfo gq_types[GQ_TYPE_COUNT] = {
    [GQ_F32]  = {"F32",  1,   4},   [GQ_F16]  = {"F16",  1,   2},   [GQ_BF16] = {"BF16", 1,   2},
    [GQ_Q4_0] = {"Q4_0", 32,  18},  [GQ_Q8_0] = {"Q8_0", 32,  34},
    [GQ_Q4_K] = {"Q4_K", 256, 144}, [GQ_Q5_K] = {"Q5_K", 256, 176}, [GQ_Q6_K] = {"Q6_K", 256, 210},
};
static inline int gq_supported(int t) { return t >= 0 && t < GQ_TYPE_COUNT && gq_types[t].name != NULL; }
static inline const char *gq_type_name(int t) { return gq_supported(t) ? gq_types[t].name : "unsupported"; }
static inline int gq_type_id(const char *name) {
    for (int t = 0; name && t < GQ_TYPE_COUNT; t++) if (gq_types[t].name && !strcmp(gq_types[t].name, name)) return t;
    return -1;
}
static inline int gq_block(int t) { return gq_supported(t) ? gq_types[t].block : 0; }
/* bytes of one row of I elements; 0 if the type is unsupported or I is not a
 * whole number of blocks */
static inline size_t gq_row_bytes(int t, int I) {
    if (!gq_supported(t) || I <= 0 || I % gq_types[t].block) return 0;
    return (size_t)(I / gq_types[t].block) * (size_t)gq_types[t].tsize;
}

/* ---- scalar conversions ------------------------------------------------------ */

static inline uint16_t gq_ld16(const uint8_t *p) { return (uint16_t)(p[0] | ((unsigned)p[1] << 8)); }
static inline void gq_st16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)(v >> 8); }
static inline float gq_bits_f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static inline uint32_t gq_f32_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static inline float gq_ld_f32(const uint8_t *p) { uint32_t b = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); return gq_bits_f32(b); }

/* IEEE half -> single, exact (subnormals, inf, NaN payload kept) */
static inline float gq_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    if (exp == 0) {
        if (man == 0) return gq_bits_f32(sign);
        int e = -1; do { e++; man <<= 1; } while (!(man & 0x400));
        return gq_bits_f32(sign | ((uint32_t)(112 - e) << 23) | ((man & 0x3FF) << 13));
    }
    if (exp == 31) return gq_bits_f32(sign | 0x7F800000u | (man << 13));
    return gq_bits_f32(sign | ((exp + 112) << 23) | (man << 13));
}
/* single -> half, round to nearest even (used by the Q8_0 join and the tests) */
static inline uint16_t gq_f32_to_f16(float f) {
    uint32_t x = gq_f32_bits(f), sign = (x >> 16) & 0x8000, man = x & 0x7FFFFF;
    int32_t e = (int32_t)((x >> 23) & 0xFF);
    if (e == 0xFF) return (uint16_t)(sign | 0x7C00 | (man ? (0x200 | (man >> 13)) : 0));
    int32_t eh = e - 112;
    if (eh >= 31) return (uint16_t)(sign | 0x7C00);
    if (eh <= 0) {
        if (eh < -10) return (uint16_t)sign;
        man |= 0x800000;
        uint32_t shift = (uint32_t)(14 - eh), half = 1u << (shift - 1), rem = man & ((1u << shift) - 1), r = man >> shift;
        if (rem > half || (rem == half && (r & 1))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = sign | ((uint32_t)eh << 10) | (man >> 13), rem = man & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return (uint16_t)r;
}
static inline float gq_bf16_to_f32(uint16_t b) { return gq_bits_f32((uint32_t)b << 16); }

/* K-quant 6-bit (scale, min) pairs: 8 pairs in 12 bytes (ggml's get_scale_min_k4) */
static inline void gq_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}
/* the eight (d*s, -(min*m)) pairs of one Q4_K/Q5_K block, decoded once per block */
static inline void gq_k4_scales(const uint8_t *sc12, float d, float min, float *d1, float *nm1) {
    for (int j = 0; j < 8; j++) { uint8_t s, m; gq_scale_min_k4(j, sc12, &s, &m); d1[j] = d * s; nm1[j] = -(min * m); }
}
/* Q6_K: element r (0..127) of one 128-element half: low 4 bits from ql, high 2
 * from qh, minus 32 */
static inline int gq_q6_get(const uint8_t *ql, const uint8_t *qh, int r) {
    int part = r >> 5, l = r & 31;
    switch (part) {
    case 0:  return (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    case 1:  return (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    case 2:  return (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
    default: return (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
    }
}

/* ---- exact dequantization (E0) -------------------------------------------------
 *
 * ggml's expression order, kept literally; the pragmas stop the compiler from
 * contracting `a*b - c` into one fma, which would change the last bit and the
 * sign of zero against llama.cpp's reference. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

static inline void gq_deq_block_q8_0(const uint8_t *b, float *y) {
    const float d = gq_f16_to_f32(gq_ld16(b));
    const int8_t *q = (const int8_t *)(b + 2);
    for (int j = 0; j < 32; j++) y[j] = q[j] * d;
}
static inline void gq_deq_block_q4_0(const uint8_t *b, float *y) {
    const float d = gq_f16_to_f32(gq_ld16(b));
    const uint8_t *q = b + 2;
    for (int j = 0; j < 16; j++) {
        const int x0 = (q[j] & 0x0F) - 8, x1 = (q[j] >> 4) - 8;
        y[j] = x0 * d; y[j + 16] = x1 * d;
    }
}
static inline void gq_deq_block_q4_k(const uint8_t *b, float *y) {
    const float d = gq_f16_to_f32(gq_ld16(b)), min = gq_f16_to_f32(gq_ld16(b + 2));
    const uint8_t *sc = b + 4, *q = b + 16;
    int is = 0; uint8_t s, m;
    for (int j = 0; j < GQ_QK_K; j += 64) {
        gq_scale_min_k4(is + 0, sc, &s, &m); const float d1 = d * s, m1 = min * m;
        gq_scale_min_k4(is + 1, sc, &s, &m); const float d2 = d * s, m2 = min * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}
static inline void gq_deq_block_q5_k(const uint8_t *b, float *y) {
    const float d = gq_f16_to_f32(gq_ld16(b)), min = gq_f16_to_f32(gq_ld16(b + 2));
    const uint8_t *sc = b + 4, *qh = b + 16, *ql = b + 48;
    int is = 0; uint8_t s, m, u1 = 1, u2 = 2;
    for (int j = 0; j < GQ_QK_K; j += 64) {
        gq_scale_min_k4(is + 0, sc, &s, &m); const float d1 = d * s, m1 = min * m;
        gq_scale_min_k4(is + 1, sc, &s, &m); const float d2 = d * s, m2 = min * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4)  + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
    }
}
static inline void gq_deq_block_q6_k(const uint8_t *b, float *y) {
    const float d = gq_f16_to_f32(gq_ld16(b + 208));
    const uint8_t *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t *)(b + 192);
    for (int n = 0; n < GQ_QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l]      = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}
/* one row of I elements (I a whole number of blocks). Returns 0, or -1 for an
 * unsupported type / bad I. */
static inline int gq_deq_row(int type, const uint8_t *row, float *y, int I) {
    if (!gq_row_bytes(type, I)) return -1;
    switch (type) {
    case GQ_F32:  for (int i = 0; i < I; i++) y[i] = gq_ld_f32(row + 4 * (size_t)i); return 0;
    case GQ_F16:  for (int i = 0; i < I; i++) y[i] = gq_f16_to_f32(gq_ld16(row + 2 * (size_t)i)); return 0;
    case GQ_BF16: for (int i = 0; i < I; i++) y[i] = gq_bf16_to_f32(gq_ld16(row + 2 * (size_t)i)); return 0;
    case GQ_Q8_0: for (int g = 0; g < I / 32; g++) gq_deq_block_q8_0(row + 34 * (size_t)g, y + 32 * g); return 0;
    case GQ_Q4_0: for (int g = 0; g < I / 32; g++) gq_deq_block_q4_0(row + 18 * (size_t)g, y + 32 * g); return 0;
    case GQ_Q4_K: for (int g = 0; g < I / 256; g++) gq_deq_block_q4_k(row + 144 * (size_t)g, y + 256 * g); return 0;
    case GQ_Q5_K: for (int g = 0; g < I / 256; g++) gq_deq_block_q5_k(row + 176 * (size_t)g, y + 256 * g); return 0;
    case GQ_Q6_K: for (int g = 0; g < I / 256; g++) gq_deq_block_q6_k(row + 210 * (size_t)g, y + 256 * g); return 0;
    default: return -1;
    }
}

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/* ---- Q8_0 <-> int8 plane + f32 scales (lossless) ------------------------------ */

/* blocks: O rows of I/32 Q8_0 blocks -> plane[O][I] int8, scales[O][I/32] f32.
 * Exactly the (q, scale, gs=32) triple matmul_q_gs consumes. */
static inline void gq_q8_0_split(const uint8_t *blocks, int I, int O, int8_t *plane, float *scales) {
    const int ng = I / 32;
    for (int o = 0; o < O; o++) {
        const uint8_t *row = blocks + (size_t)o * ng * 34;
        for (int g = 0; g < ng; g++) {
            scales[(size_t)o * ng + g] = gq_f16_to_f32(gq_ld16(row + 34 * (size_t)g));
            memcpy(plane + (size_t)o * I + 32 * (size_t)g, row + 34 * (size_t)g + 2, 32);
        }
    }
}
/* inverse, for the round-trip test (scales came from f16, so they convert back exactly) */
static inline void gq_q8_0_join(const int8_t *plane, const float *scales, int I, int O, uint8_t *blocks) {
    const int ng = I / 32;
    for (int o = 0; o < O; o++) {
        uint8_t *row = blocks + (size_t)o * ng * 34;
        for (int g = 0; g < ng; g++) {
            gq_st16(row + 34 * (size_t)g, gq_f32_to_f16(scales[(size_t)o * ng + g]));
            memcpy(row + 34 * (size_t)g + 2, plane + (size_t)o * I + 32 * (size_t)g, 32);
        }
    }
}

/* ---- scalar reference dots (the numerics contract) ---------------------------- */

static inline float gq_hsum8_scalar(const float *l) {
    float a0 = l[0] + l[4], a1 = l[1] + l[5], a2 = l[2] + l[6], a3 = l[3] + l[7];
    float b0 = a0 + a2, b1 = a1 + a3;
    return b0 + b1;
}
/* 32 quantized integers v[32] against xb[32]: lane[l] = fma over elements l, l+8, l+16, l+24 */
static inline void gq_lane32_ref(float *lane, const int *v, const float *xb) {
    for (int l = 0; l < 8; l++) lane[l] = 0.f;
    for (int i = 0; i < 32; i++) lane[i & 7] = fmaf((float)v[i], xb[i], lane[i & 7]);
}
/* lane sums of the activations themselves (K-quant min terms): x[l] + x[l+8] + x[l+16] + x[l+24] */
static inline void gq_xlane32_ref(float *lx, const float *xb) {
    for (int l = 0; l < 8; l++) { lx[l] = xb[l]; lx[l] += xb[l + 8]; lx[l] += xb[l + 16]; lx[l] += xb[l + 24]; }
}

/* the K-quant min terms need the sum of x over each 32-element sub-block
 * (-(min*m) * sum x). It depends on the activation only, so a caller with
 * many rows against one x (gq_matmul, gq_moe_run) computes the I/32 sums once
 * with gq_xsum32 and passes `xs`; a NULL xs makes the kernels compute them
 * inline. Both ways perform the same adds in the same order (8 lane sums, then
 * the fixed 8->1 tree), so results are identical. */
static inline float gq_xsub32_ref(const float *xb) { float lx[8]; gq_xlane32_ref(lx, xb); return gq_hsum8_scalar(lx); }
static inline void gq_xsum32_ref(const float *x, int I, float *xs) {
    for (int g = 0; g < I / 32; g++) xs[g] = gq_xsub32_ref(x + 32 * g);
}
static inline float gq_dot_f32_ref(const uint8_t *w, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < I; i++) acc[i & 7] = fmaf(gq_ld_f32(w + 4 * (size_t)i), x[i], acc[i & 7]);
    return gq_hsum8_scalar(acc);
}
static inline float gq_dot_f16_ref(const uint8_t *w, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < I; i++) acc[i & 7] = fmaf(gq_f16_to_f32(gq_ld16(w + 2 * (size_t)i)), x[i], acc[i & 7]);
    return gq_hsum8_scalar(acc);
}
static inline float gq_dot_bf16_ref(const uint8_t *w, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < I; i++) acc[i & 7] = fmaf(gq_bf16_to_f32(gq_ld16(w + 2 * (size_t)i)), x[i], acc[i & 7]);
    return gq_hsum8_scalar(acc);
}
static inline float gq_dot_q8_0_ref(const uint8_t *w, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 34 * (size_t)g; const float d = gq_f16_to_f32(gq_ld16(b));
        int v[32]; for (int i = 0; i < 32; i++) v[i] = (int8_t)b[2 + i];
        float lane[8]; gq_lane32_ref(lane, v, x + 32 * g);
        for (int l = 0; l < 8; l++) acc[l] = fmaf(lane[l], d, acc[l]);
    }
    return gq_hsum8_scalar(acc);
}
static inline float gq_dot_q4_0_ref(const uint8_t *w, const float *x, int I) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 18 * (size_t)g; const float d = gq_f16_to_f32(gq_ld16(b));
        int v[32]; for (int i = 0; i < 16; i++) { v[i] = (b[2 + i] & 0xF) - 8; v[i + 16] = (b[2 + i] >> 4) - 8; }
        float lane[8]; gq_lane32_ref(lane, v, x + 32 * g);
        for (int l = 0; l < 8; l++) acc[l] = fmaf(lane[l], d, acc[l]);
    }
    return gq_hsum8_scalar(acc);
}
/* Q4_K / Q5_K: per 32-element sub-block j with (d1, m1): acc += lane_q*d1; acc += lane_x*(-m1) */
static inline float gq_dot_q45_k_ref(const uint8_t *w, const float *x, const float *xs, int I, int five) {
    /* q terms: two 8-lane accumulators (low / high nibble sub-blocks) keep the
     * fma chain short; min terms: one accumulator whose 8 lanes are the 8
     * sub-blocks of a block (one fma per block). Combined once at the end. */
    const size_t bs = five ? 176 : 144;
    float accq[2][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}}, accx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + bs * (size_t)g;
        const uint8_t *qh = b + 16, *ql = five ? b + 48 : b + 16;
        const float *xb = x + GQ_QK_K * g;
        float d1[8], nm1[8]; gq_k4_scales(b + 4, gq_f16_to_f32(gq_ld16(b)), gq_f16_to_f32(gq_ld16(b + 2)), d1, nm1);
        for (int j = 0; j < 4; j++) for (int half = 0; half < 2; half++) {
            int v[32]; float lane[8]; const int k = 2 * j + half;
            const uint8_t u = (uint8_t)((half ? 2u : 1u) << (2 * j));
            for (int l = 0; l < 32; l++) v[l] = (half ? (ql[32 * j + l] >> 4) : (ql[32 * j + l] & 0xF)) + (five && (qh[l] & u) ? 16 : 0);
            gq_lane32_ref(lane, v, xb + 32 * k);
            for (int l = 0; l < 8; l++) accq[half][l] = fmaf(lane[l], d1[k], accq[half][l]);
            const float xsub = xs ? xs[8 * g + k] : gq_xsub32_ref(xb + 32 * k);
            accx[k] = fmaf(xsub, nm1[k], accx[k]);
        }
    }
    float tq[8]; for (int l = 0; l < 8; l++) tq[l] = accq[0][l] + accq[1][l];
    return gq_hsum8_scalar(tq) + gq_hsum8_scalar(accx);
}
static inline float gq_dot_q4_k_ref(const uint8_t *w, const float *x, const float *xs, int I) { return gq_dot_q45_k_ref(w, x, xs, I, 0); }
static inline float gq_dot_q5_k_ref(const uint8_t *w, const float *x, const float *xs, int I) { return gq_dot_q45_k_ref(w, x, xs, I, 1); }
/* Q6_K: per 16-element sub-block k with int8 scale: accb += lane*sc[k]; per block: acc += accb*d */
static inline float gq_dot_q6_k_ref(const uint8_t *w, const float *x, int I) {
    float acc[4][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + 210 * (size_t)g; const float d = gq_f16_to_f32(gq_ld16(b + 208));
        float accb[4][8] = {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = b + 64 * h, *qh = b + 128 + 32 * h; const int8_t *sc = (const int8_t *)(b + 192 + 8 * h);
            const float *xh = x + GQ_QK_K * g + 128 * h;
            for (int k = 0; k < 8; k++) {
                float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int t = 0; t < 16; t++) { int r = 16 * k + t; lane[t & 7] = fmaf((float)gq_q6_get(ql, qh, r), xh[r], lane[t & 7]); }
                for (int l = 0; l < 8; l++) accb[k & 3][l] = fmaf(lane[l], (float)sc[k], accb[k & 3][l]);
            }
        }
        for (int i = 0; i < 4; i++) for (int l = 0; l < 8; l++) acc[i][l] = fmaf(accb[i][l], d, acc[i][l]);
    }
    float t[8]; for (int l = 0; l < 8; l++) t[l] = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    return gq_hsum8_scalar(t);
}

/* ---- AVX2 -------------------------------------------------------------------- */
#ifdef GQ_HAVE_AVX2
static inline float gq_hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    __m128 a = _mm_add_ps(lo, hi);
    __m128 b = _mm_add_ps(a, _mm_movehl_ps(a, a));
    __m128 c = _mm_add_ss(b, _mm_shuffle_ps(b, b, 1));
    return _mm_cvtss_f32(c);
}
static inline __m256 gq_cvt8(__m128i v) { return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)); }
/* 32 int8 values v against xb[32] -> the 8 lane sums, element i in lane i&7 */
static inline __m256 gq_lane32_avx2(__m256i v, const float *xb) {
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    __m256 lane = _mm256_mul_ps(gq_cvt8(lo), _mm256_loadu_ps(xb));
    lane = _mm256_fmadd_ps(gq_cvt8(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(xb + 8),  lane);
    lane = _mm256_fmadd_ps(gq_cvt8(hi),                    _mm256_loadu_ps(xb + 16), lane);
    lane = _mm256_fmadd_ps(gq_cvt8(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(xb + 24), lane);
    return lane;
}
/* first 16 of the 32 values (two octets) only */
static inline __m256 gq_lane16_avx2(__m128i v16, const float *xb) {
    __m256 lane = _mm256_mul_ps(gq_cvt8(v16), _mm256_loadu_ps(xb));
    return _mm256_fmadd_ps(gq_cvt8(_mm_srli_si128(v16, 8)), _mm256_loadu_ps(xb + 8), lane);
}
static inline __m256 gq_xlane32_avx2(const float *xb) {
    __m256 lx = _mm256_loadu_ps(xb);
    lx = _mm256_add_ps(lx, _mm256_loadu_ps(xb + 8));
    lx = _mm256_add_ps(lx, _mm256_loadu_ps(xb + 16));
    return _mm256_add_ps(lx, _mm256_loadu_ps(xb + 24));
}
static inline float gq_dot_f32_avx2(const uint8_t *w, const float *x, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < I; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps((const float *)(w + 4 * (size_t)i)), _mm256_loadu_ps(x + i), acc);
    return gq_hsum8(acc);
}
static inline float gq_dot_f16_avx2(const uint8_t *w, const float *x, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < I; i += 8) {
#if defined(__F16C__)
        __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + 2 * (size_t)i)));
#else
        float t[8]; for (int k = 0; k < 8; k++) t[k] = gq_f16_to_f32(gq_ld16(w + 2 * (size_t)(i + k)));
        __m256 wv = _mm256_loadu_ps(t);
#endif
        acc = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + i), acc);
    }
    return gq_hsum8(acc);
}
static inline float gq_dot_bf16_avx2(const uint8_t *w, const float *x, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < I; i += 8) {
        __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + 2 * (size_t)i)));
        acc = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(u, 16)), _mm256_loadu_ps(x + i), acc);
    }
    return gq_hsum8(acc);
}
static inline float gq_dot_q8_0_avx2(const uint8_t *w, const float *x, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 34 * (size_t)g;
        __m256 lane = gq_lane32_avx2(_mm256_loadu_si256((const __m256i *)(b + 2)), x + 32 * g);
        acc = _mm256_fmadd_ps(lane, _mm256_set1_ps(gq_f16_to_f32(gq_ld16(b))), acc);
    }
    return gq_hsum8(acc);
}
static inline float gq_dot_q4_0_avx2(const uint8_t *w, const float *x, int I) {
    const __m128i m4 = _mm_set1_epi8(0x0F), e8 = _mm_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 18 * (size_t)g;
        __m128i q = _mm_loadu_si128((const __m128i *)(b + 2));
        __m128i lo = _mm_sub_epi8(_mm_and_si128(q, m4), e8), hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), m4), e8);
        __m256 lane = gq_lane32_avx2(_mm256_set_m128i(hi, lo), x + 32 * g);
        acc = _mm256_fmadd_ps(lane, _mm256_set1_ps(gq_f16_to_f32(gq_ld16(b))), acc);
    }
    return gq_hsum8(acc);
}
static inline void gq_xsum32_avx2(const float *x, int I, float *xs) {
    for (int g = 0; g < I / 32; g++) xs[g] = gq_hsum8(gq_xlane32_avx2(x + 32 * g));
}
/* the eight (d*s, -(min*m)) pairs of a block with vector ops; same values as gq_k4_scales */
static inline void gq_k4_scales_avx2(const uint8_t *sc, float d, float min, float *d1, float *nm1) {
    uint32_t a, b, c; memcpy(&a, sc, 4); memcpy(&b, sc + 4, 4); memcpy(&c, sc + 8, 4);
    uint32_t s_lo = a & 0x3F3F3F3Fu, m_lo = b & 0x3F3F3F3Fu;
    uint32_t s_hi = (c & 0x0F0F0F0Fu) | (((a >> 6) & 0x03030303u) << 4);
    uint32_t m_hi = ((c >> 4) & 0x0F0F0F0Fu) | (((b >> 6) & 0x03030303u) << 4);
    __m256 sf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_set_epi32(0, 0, (int)s_hi, (int)s_lo)));
    __m256 mf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_set_epi32(0, 0, (int)m_hi, (int)m_lo)));
    _mm256_storeu_ps(d1, _mm256_mul_ps(sf, _mm256_set1_ps(d)));
    _mm256_storeu_ps(nm1, _mm256_xor_ps(_mm256_mul_ps(mf, _mm256_set1_ps(min)), _mm256_set1_ps(-0.f)));
}
static inline float gq_dot_q45_k_avx2(const uint8_t *w, const float *x, const float *xs, int I, int five) {
    const size_t bs = five ? 176 : 144;
    const __m256i m4 = _mm256_set1_epi8(0x0F), c16 = _mm256_set1_epi8(16), z = _mm256_setzero_si256();
    __m256 accq0 = _mm256_setzero_ps(), accq1 = _mm256_setzero_ps(), accx = _mm256_setzero_ps();
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + bs * (size_t)g;
        const uint8_t *ql = five ? b + 48 : b + 16;
        const float *xb = x + GQ_QK_K * g;
        float d1[8], nm1[8]; gq_k4_scales_avx2(b + 4, gq_f16_to_f32(gq_ld16(b)), gq_f16_to_f32(gq_ld16(b + 2)), d1, nm1);
        __m256i qh = five ? _mm256_loadu_si256((const __m256i *)(b + 16)) : z;
        for (int j = 0; j < 4; j++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(ql + 32 * j));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if (five) {
                __m256i u1 = _mm256_set1_epi8((char)(1u << (2 * j))), u2 = _mm256_set1_epi8((char)(2u << (2 * j)));
                lo = _mm256_add_epi8(lo, _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u1), z), c16));
                hi = _mm256_add_epi8(hi, _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u2), z), c16));
            }
            accq0 = _mm256_fmadd_ps(gq_lane32_avx2(lo, xb + 64 * j),      _mm256_set1_ps(d1[2 * j]),     accq0);
            accq1 = _mm256_fmadd_ps(gq_lane32_avx2(hi, xb + 64 * j + 32), _mm256_set1_ps(d1[2 * j + 1]), accq1);
        }
        __m256 xsub;
        if (xs) xsub = _mm256_loadu_ps(xs + 8 * g);
        else { float t[8]; for (int k = 0; k < 8; k++) t[k] = gq_hsum8(gq_xlane32_avx2(xb + 32 * k)); xsub = _mm256_loadu_ps(t); }
        accx = _mm256_fmadd_ps(xsub, _mm256_loadu_ps(nm1), accx);
    }
    return gq_hsum8(_mm256_add_ps(accq0, accq1)) + gq_hsum8(accx);
}
static inline float gq_dot_q4_k_avx2(const uint8_t *w, const float *x, const float *xs, int I) { return gq_dot_q45_k_avx2(w, x, xs, I, 0); }
static inline float gq_dot_q5_k_avx2(const uint8_t *w, const float *x, const float *xs, int I) { return gq_dot_q45_k_avx2(w, x, xs, I, 1); }
static inline float gq_dot_q6_k_avx2(const uint8_t *w, const float *x, int I) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3), c32 = _mm256_set1_epi8(32);
    __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + 210 * (size_t)g; const __m256 d = _mm256_set1_ps(gq_f16_to_f32(gq_ld16(b + 208)));
        __m256 accb[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = b + 64 * h, *qh = b + 128 + 32 * h; const int8_t *sc = (const int8_t *)(b + 192 + 8 * h);
            const float *xh = x + GQ_QK_K * g + 128 * h;
            __m256i q0 = _mm256_loadu_si256((const __m256i *)ql), q1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hv = _mm256_loadu_si256((const __m256i *)qh);
            __m256i p[4];
            p[0] = _mm256_or_si256(_mm256_and_si256(q0, m4),                       _mm256_slli_epi16(_mm256_and_si256(hv, m3), 4));
            p[1] = _mm256_or_si256(_mm256_and_si256(q1, m4),                       _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 2), m3), 4));
            p[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 4), m3), 4));
            p[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 6), m3), 4));
            for (int part = 0; part < 4; part++) {
                __m256i v = _mm256_sub_epi8(p[part], c32);
                __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
                const int k0 = (2 * part) & 3, k1 = (2 * part + 1) & 3;
                accb[k0] = _mm256_fmadd_ps(gq_lane16_avx2(lo, xh + 32 * part),      _mm256_set1_ps((float)sc[2 * part]),     accb[k0]);
                accb[k1] = _mm256_fmadd_ps(gq_lane16_avx2(hi, xh + 32 * part + 16), _mm256_set1_ps((float)sc[2 * part + 1]), accb[k1]);
            }
        }
        for (int i = 0; i < 4; i++) acc[i] = _mm256_fmadd_ps(accb[i], d, acc[i]);
    }
    return gq_hsum8(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
}
#endif /* GQ_HAVE_AVX2 */

/* ---- NEON (AArch64): the 8 lanes are two float32x4, lanes 0-3 and 4-7 ---------- */
#ifdef GQ_HAVE_NEON
typedef struct { float32x4_t lo, hi; } gq_v8;
static inline gq_v8 gq_v8_zero(void) { gq_v8 r = { vdupq_n_f32(0.f), vdupq_n_f32(0.f) }; return r; }
static inline float gq_v8_hsum(gq_v8 v) { float l[8]; vst1q_f32(l, v.lo); vst1q_f32(l + 4, v.hi); return gq_hsum8_scalar(l); }
/* 8 int8 -> 8 f32 (two quads) */
static inline void gq_cvt8_neon(int8x8_t v, float32x4_t *a, float32x4_t *b) {
    int16x8_t w = vmovl_s8(v);
    *a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))); *b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
}
/* acc += v8 * x8 (fused), v given as 8 int8 */
static inline gq_v8 gq_fma8_neon(gq_v8 acc, int8x8_t v, const float *x) {
    float32x4_t a, b; gq_cvt8_neon(v, &a, &b);
    acc.lo = vfmaq_f32(acc.lo, a, vld1q_f32(x)); acc.hi = vfmaq_f32(acc.hi, b, vld1q_f32(x + 4));
    return acc;
}
/* first product without a prior accumulator (matches fmaf(v, x, 0) = round(v*x)) */
static inline gq_v8 gq_mul8_neon(int8x8_t v, const float *x) {
    float32x4_t a, b; gq_cvt8_neon(v, &a, &b);
    gq_v8 r = { vmulq_f32(a, vld1q_f32(x)), vmulq_f32(b, vld1q_f32(x + 4)) }; return r;
}
static inline gq_v8 gq_lane32_neon(int8x16_t v0, int8x16_t v1, const float *xb) {
    gq_v8 lane = gq_mul8_neon(vget_low_s8(v0), xb);
    lane = gq_fma8_neon(lane, vget_high_s8(v0), xb + 8);
    lane = gq_fma8_neon(lane, vget_low_s8(v1), xb + 16);
    return gq_fma8_neon(lane, vget_high_s8(v1), xb + 24);
}
static inline gq_v8 gq_lane16_neon(int8x16_t v, const float *xb) {
    gq_v8 lane = gq_mul8_neon(vget_low_s8(v), xb);
    return gq_fma8_neon(lane, vget_high_s8(v), xb + 8);
}
static inline gq_v8 gq_xlane32_neon(const float *xb) {
    gq_v8 r = { vld1q_f32(xb), vld1q_f32(xb + 4) };
    r.lo = vaddq_f32(r.lo, vld1q_f32(xb + 8));  r.hi = vaddq_f32(r.hi, vld1q_f32(xb + 12));
    r.lo = vaddq_f32(r.lo, vld1q_f32(xb + 16)); r.hi = vaddq_f32(r.hi, vld1q_f32(xb + 20));
    r.lo = vaddq_f32(r.lo, vld1q_f32(xb + 24)); r.hi = vaddq_f32(r.hi, vld1q_f32(xb + 28));
    return r;
}
static inline gq_v8 gq_fold_neon(gq_v8 acc, gq_v8 lane, float s) {
    acc.lo = vfmaq_n_f32(acc.lo, lane.lo, s); acc.hi = vfmaq_n_f32(acc.hi, lane.hi, s); return acc;
}
static inline float gq_dot_f32_neon(const uint8_t *w, const float *x, int I) {
    gq_v8 acc = gq_v8_zero();
    for (int i = 0; i < I; i += 8) {
        acc.lo = vfmaq_f32(acc.lo, vld1q_f32((const float *)(w + 4 * (size_t)i)),       vld1q_f32(x + i));
        acc.hi = vfmaq_f32(acc.hi, vld1q_f32((const float *)(w + 4 * (size_t)i + 16)),  vld1q_f32(x + i + 4));
    }
    return gq_v8_hsum(acc);
}
static inline float gq_dot_f16_neon(const uint8_t *w, const float *x, int I) {
    gq_v8 acc = gq_v8_zero();
    for (int i = 0; i < I; i += 8) {
        float32x4_t a = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *)(w + 2 * (size_t)i))));
        float32x4_t b = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *)(w + 2 * (size_t)i + 8))));
        acc.lo = vfmaq_f32(acc.lo, a, vld1q_f32(x + i)); acc.hi = vfmaq_f32(acc.hi, b, vld1q_f32(x + i + 4));
    }
    return gq_v8_hsum(acc);
}
static inline float gq_dot_bf16_neon(const uint8_t *w, const float *x, int I) {
    gq_v8 acc = gq_v8_zero();
    for (int i = 0; i < I; i += 8) {
        uint16x8_t u = vld1q_u16((const uint16_t *)(w + 2 * (size_t)i));
        float32x4_t a = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16));
        float32x4_t b = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16));
        acc.lo = vfmaq_f32(acc.lo, a, vld1q_f32(x + i)); acc.hi = vfmaq_f32(acc.hi, b, vld1q_f32(x + i + 4));
    }
    return gq_v8_hsum(acc);
}
static inline float gq_dot_q8_0_neon(const uint8_t *w, const float *x, int I) {
    gq_v8 acc = gq_v8_zero();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 34 * (size_t)g;
        gq_v8 lane = gq_lane32_neon(vld1q_s8((const int8_t *)(b + 2)), vld1q_s8((const int8_t *)(b + 18)), x + 32 * g);
        acc = gq_fold_neon(acc, lane, gq_f16_to_f32(gq_ld16(b)));
    }
    return gq_v8_hsum(acc);
}
static inline float gq_dot_q4_0_neon(const uint8_t *w, const float *x, int I) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F); const int8x16_t e8 = vdupq_n_s8(8);
    gq_v8 acc = gq_v8_zero();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 18 * (size_t)g;
        uint8x16_t q = vld1q_u8(b + 2);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m4)), e8), hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), e8);
        acc = gq_fold_neon(acc, gq_lane32_neon(lo, hi, x + 32 * g), gq_f16_to_f32(gq_ld16(b)));
    }
    return gq_v8_hsum(acc);
}
static inline void gq_xsum32_neon(const float *x, int I, float *xs) {
    for (int g = 0; g < I / 32; g++) xs[g] = gq_v8_hsum(gq_xlane32_neon(x + 32 * g));
}
static inline gq_v8 gq_v8_load(const float *p) { gq_v8 r = { vld1q_f32(p), vld1q_f32(p + 4) }; return r; }
static inline gq_v8 gq_v8_add(gq_v8 a, gq_v8 b) { gq_v8 r = { vaddq_f32(a.lo, b.lo), vaddq_f32(a.hi, b.hi) }; return r; }
static inline float gq_dot_q45_k_neon(const uint8_t *w, const float *x, const float *xs, int I, int five) {
    const size_t bs = five ? 176 : 144;
    const uint8x16_t m4 = vdupq_n_u8(0x0F), c16 = vdupq_n_u8(16);
    gq_v8 accq0 = gq_v8_zero(), accq1 = gq_v8_zero(), accx = gq_v8_zero();
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + bs * (size_t)g;
        const uint8_t *ql = five ? b + 48 : b + 16;
        const float *xb = x + GQ_QK_K * g;
        float d1[8], nm1[8]; gq_k4_scales(b + 4, gq_f16_to_f32(gq_ld16(b)), gq_f16_to_f32(gq_ld16(b + 2)), d1, nm1);
        uint8x16_t qh0 = five ? vld1q_u8(b + 16) : vdupq_n_u8(0), qh1 = five ? vld1q_u8(b + 32) : vdupq_n_u8(0);
        for (int j = 0; j < 4; j++) {
            uint8x16_t q0 = vld1q_u8(ql + 32 * j), q1 = vld1q_u8(ql + 32 * j + 16);
            uint8x16_t lo0 = vandq_u8(q0, m4), lo1 = vandq_u8(q1, m4), hi0 = vshrq_n_u8(q0, 4), hi1 = vshrq_n_u8(q1, 4);
            if (five) {
                uint8x16_t u1 = vdupq_n_u8((uint8_t)(1u << (2 * j))), u2 = vdupq_n_u8((uint8_t)(2u << (2 * j)));
                lo0 = vaddq_u8(lo0, vandq_u8(vtstq_u8(qh0, u1), c16)); lo1 = vaddq_u8(lo1, vandq_u8(vtstq_u8(qh1, u1), c16));
                hi0 = vaddq_u8(hi0, vandq_u8(vtstq_u8(qh0, u2), c16)); hi1 = vaddq_u8(hi1, vandq_u8(vtstq_u8(qh1, u2), c16));
            }
            accq0 = gq_fold_neon(accq0, gq_lane32_neon(vreinterpretq_s8_u8(lo0), vreinterpretq_s8_u8(lo1), xb + 64 * j), d1[2 * j]);
            accq1 = gq_fold_neon(accq1, gq_lane32_neon(vreinterpretq_s8_u8(hi0), vreinterpretq_s8_u8(hi1), xb + 64 * j + 32), d1[2 * j + 1]);
        }
        float t[8];
        if (xs) memcpy(t, xs + 8 * g, sizeof t); else for (int k = 0; k < 8; k++) t[k] = gq_v8_hsum(gq_xlane32_neon(xb + 32 * k));
        accx.lo = vfmaq_f32(accx.lo, vld1q_f32(t), vld1q_f32(nm1)); accx.hi = vfmaq_f32(accx.hi, vld1q_f32(t + 4), vld1q_f32(nm1 + 4));
    }
    return gq_v8_hsum(gq_v8_add(accq0, accq1)) + gq_v8_hsum(accx);
}
static inline float gq_dot_q4_k_neon(const uint8_t *w, const float *x, const float *xs, int I) { return gq_dot_q45_k_neon(w, x, xs, I, 0); }
static inline float gq_dot_q5_k_neon(const uint8_t *w, const float *x, const float *xs, int I) { return gq_dot_q45_k_neon(w, x, xs, I, 1); }
static inline float gq_dot_q6_k_neon(const uint8_t *w, const float *x, int I) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3); const int8x16_t c32 = vdupq_n_s8(32);
    gq_v8 acc[4] = { gq_v8_zero(), gq_v8_zero(), gq_v8_zero(), gq_v8_zero() };
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + 210 * (size_t)g; const float d = gq_f16_to_f32(gq_ld16(b + 208));
        gq_v8 accb[4] = { gq_v8_zero(), gq_v8_zero(), gq_v8_zero(), gq_v8_zero() };
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = b + 64 * h, *qh = b + 128 + 32 * h; const int8_t *sc = (const int8_t *)(b + 192 + 8 * h);
            const float *xh = x + GQ_QK_K * g + 128 * h;
            for (int part = 0; part < 4; part++) {
                const uint8_t *qlp = ql + (part & 1 ? 32 : 0);
                for (int half = 0; half < 2; half++) {
                    uint8x16_t qv = vld1q_u8(qlp + 16 * half), hv = vld1q_u8(qh + 16 * half);
                    uint8x16_t base = part < 2 ? vandq_u8(qv, m4) : vshrq_n_u8(qv, 4);
                    uint8x16_t hb;
                    switch (part) {
                    case 0:  hb = vandq_u8(hv, m3); break;
                    case 1:  hb = vandq_u8(vshrq_n_u8(hv, 2), m3); break;
                    case 2:  hb = vandq_u8(vshrq_n_u8(hv, 4), m3); break;
                    default: hb = vshrq_n_u8(hv, 6); break;
                    }
                    int8x16_t v = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(base, vshlq_n_u8(hb, 4))), c32);
                    const int k = (2 * part + half) & 3;
                    accb[k] = gq_fold_neon(accb[k], gq_lane16_neon(v, xh + 32 * part + 16 * half), (float)sc[2 * part + half]);
                }
            }
        }
        for (int i = 0; i < 4; i++) acc[i] = gq_fold_neon(acc[i], accb[i], d);
    }
    return gq_v8_hsum(gq_v8_add(gq_v8_add(acc[0], acc[1]), gq_v8_add(acc[2], acc[3])));
}
#endif /* GQ_HAVE_NEON */

/* ---- dispatch -------------------------------------------------------------------- */

#if defined(GQ_HAVE_AVX2)
#define GQ_PICK(name, w, x, I) ((I) % 32 ? gq_dot_##name##_ref(w, x, I) : gq_dot_##name##_avx2(w, x, I))
#define GQ_PICKX(name, w, x, xs, I) ((I) % 32 ? gq_dot_##name##_ref(w, x, xs, I) : gq_dot_##name##_avx2(w, x, xs, I))
#define GQ_XSUM(x, I, xs) ((I) % 32 ? gq_xsum32_ref(x, I, xs) : gq_xsum32_avx2(x, I, xs))
#define GQ_PATH "avx2"
#elif defined(GQ_HAVE_NEON)
#define GQ_PICK(name, w, x, I) ((I) % 32 ? gq_dot_##name##_ref(w, x, I) : gq_dot_##name##_neon(w, x, I))
#define GQ_PICKX(name, w, x, xs, I) ((I) % 32 ? gq_dot_##name##_ref(w, x, xs, I) : gq_dot_##name##_neon(w, x, xs, I))
#define GQ_XSUM(x, I, xs) ((I) % 32 ? gq_xsum32_ref(x, I, xs) : gq_xsum32_neon(x, I, xs))
#define GQ_PATH "neon"
#else
#define GQ_PICK(name, w, x, I) gq_dot_##name##_ref(w, x, I)
#define GQ_PICKX(name, w, x, xs, I) gq_dot_##name##_ref(w, x, xs, I)
#define GQ_XSUM(x, I, xs) gq_xsum32_ref(x, I, xs)
#define GQ_PATH "scalar"
#endif
static inline float gq_dot_f32(const uint8_t *w, const float *x, int I)  { return GQ_PICK(f32, w, x, I); }
static inline float gq_dot_f16(const uint8_t *w, const float *x, int I)  { return GQ_PICK(f16, w, x, I); }
static inline float gq_dot_bf16(const uint8_t *w, const float *x, int I) { return GQ_PICK(bf16, w, x, I); }
static inline float gq_dot_q4_0(const uint8_t *w, const float *x, int I) { return GQ_PICK(q4_0, w, x, I); }
static inline float gq_dot_q8_0(const uint8_t *w, const float *x, int I) { return GQ_PICK(q8_0, w, x, I); }
static inline float gq_dot_q4_k(const uint8_t *w, const float *x, const float *xs, int I) { return GQ_PICKX(q4_k, w, x, xs, I); }
static inline float gq_dot_q5_k(const uint8_t *w, const float *x, const float *xs, int I) { return GQ_PICKX(q5_k, w, x, xs, I); }
static inline float gq_dot_q6_k(const uint8_t *w, const float *x, int I) { return GQ_PICK(q6_k, w, x, I); }
/* per-32 activation sums for the K-quant min terms, xs[I/32]; see gq_xsum32_ref */
static inline void gq_xsum32(const float *x, int I, float *xs) { GQ_XSUM(x, I, xs); }
#undef GQ_PICK
#undef GQ_PICKX
#undef GQ_XSUM

/* scalar reference by type (tests); xs optional, only Q4_K/Q5_K read it */
static inline float gq_dot_row_ref_xs(int type, const uint8_t *row, const float *x, const float *xs, int I) {
    switch (type) {
    case GQ_F32:  return gq_dot_f32_ref(row, x, I);   case GQ_F16:  return gq_dot_f16_ref(row, x, I);
    case GQ_BF16: return gq_dot_bf16_ref(row, x, I);  case GQ_Q4_0: return gq_dot_q4_0_ref(row, x, I);
    case GQ_Q8_0: return gq_dot_q8_0_ref(row, x, I);  case GQ_Q4_K: return gq_dot_q4_k_ref(row, x, xs, I);
    case GQ_Q5_K: return gq_dot_q5_k_ref(row, x, xs, I);  case GQ_Q6_K: return gq_dot_q6_k_ref(row, x, I);
    default: return 0.f;
    }
}
static inline float gq_dot_row_ref(int type, const uint8_t *row, const float *x, int I) { return gq_dot_row_ref_xs(type, row, x, NULL, I); }
/* fastest path by type; I must be a whole number of blocks (the caller checked
 * gq_row_bytes); xs from gq_xsum32(x) or NULL */
static inline float gq_dot_row_xs(int type, const uint8_t *row, const float *x, const float *xs, int I) {
    switch (type) {
    case GQ_F32:  return gq_dot_f32(row, x, I);   case GQ_F16:  return gq_dot_f16(row, x, I);
    case GQ_BF16: return gq_dot_bf16(row, x, I);  case GQ_Q4_0: return gq_dot_q4_0(row, x, I);
    case GQ_Q8_0: return gq_dot_q8_0(row, x, I);  case GQ_Q4_K: return gq_dot_q4_k(row, x, xs, I);
    case GQ_Q5_K: return gq_dot_q5_k(row, x, xs, I);  case GQ_Q6_K: return gq_dot_q6_k(row, x, I);
    default: return 0.f;
    }
}
static inline float gq_dot_row(int type, const uint8_t *row, const float *x, int I) { return gq_dot_row_xs(type, row, x, NULL, I); }
static inline int gq_needs_xs(int type) { return type == GQ_Q4_K || type == GQ_Q5_K; }
#define GQ_XS_MAX_I 16384

/* ---- dense ------------------------------------------------------------------------- */

/* y[O] = W[O][I] . x[I], W as raw blocks of `type`. OMP over rows (the lm_head
 * call: 248 320 rows). Returns 0, -1 for an unsupported type / bad shape. */
static inline int gq_matmul(float *y, const float *x, int type, const uint8_t *w, int I, int O) {
    const size_t rb = gq_row_bytes(type, I);
    if (!rb || O <= 0) return -1;
    float xsb[GQ_XS_MAX_I / 32]; const float *xs = NULL;
    if (gq_needs_xs(type) && I <= GQ_XS_MAX_I) { gq_xsum32(x, I, xsb); xs = xsb; }
    #pragma omp parallel for schedule(static) if(O >= 256)
    for (int o = 0; o < O; o++) y[o] = gq_dot_row_xs(type, w + rb * (size_t)o, x, xs, I);
    return 0;
}
/* one row (token_embd lookup), exact dequant */
static inline int gq_embed_row(int type, const uint8_t *w, int64_t row, float *out, int I) {
    const size_t rb = gq_row_bytes(type, I);
    if (!rb || row < 0) return -1;
    return gq_deq_row(type, w + rb * (size_t)row, out, I);
}

/* ---- one layer: the K-quant twin of xf_moe_run ----------------------------------------
 *
 * S tokens, K routed experts each. idx[s*K+k] is the expert id (or -1), val the
 * router weight, experts[s*K+k] the resident expert (NULL iff idx < 0). An
 * expert is three raw matrices: gate/up [F][H] of types tg/tu, down [H][F] of
 * type td. out[s][H] = sum_k val[s][k] * expert_k(x[s]) accumulated in k order,
 * so the result does not depend on the thread count. f32 activations. */
typedef struct {
    const uint8_t *g, *u, *d;
    int tg, tu, td;
} GqExpert;

static inline size_t gq_moe_scratch_bytes(int S, int K, int H, int F) {
    size_t n = (size_t)S * K;
    return n * (2 * (size_t)F + (size_t)H) * sizeof(float)   /* g, u per (s,k); contribution per (s,k) */
         + n * (size_t)F * sizeof(float)                       /* h per (s,k) */
         + ((size_t)S * (H / 32) + n * (F / 32)) * sizeof(float) /* per-32 sums of x and h (K-quant min terms) */
         + n * sizeof(int) * 4 + 8192;
}
static inline void gq_swiglu(float *h, const float *g, const float *u, int F) {
    for (int i = 0; i < F; i++) { float gv = g[i]; h[i] = (gv / (1.f + expf(-gv))) * u[i]; }
}
static inline void gq_moe_run(float *out, const float *x, int S, int K, int H, int F,
                              const int *idx, const float *val, const GqExpert *const *experts, void *scratch) {
    const size_t n = (size_t)S * K;
    char *p = (char *)scratch;
#define GQ_TAKE(T, count) ((T *)p); p += (((size_t)(count) * sizeof(T)) + 63) & ~(size_t)63
    float *g   = GQ_TAKE(float, n * F);
    float *u   = GQ_TAKE(float, n * F);
    float *h   = GQ_TAKE(float, n * F);
    float *ctb = GQ_TAKE(float, n * H);
    float *xs  = GQ_TAKE(float, (size_t)S * (H / 32));
    float *hs  = GQ_TAKE(float, n * (F / 32));
    int *uniq  = GQ_TAKE(int, n);
    int *head  = GQ_TAKE(int, n);
    int *next  = GQ_TAKE(int, n);
    int *cnt   = GQ_TAKE(int, n);
#undef GQ_TAKE

    int nu = 0;
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k; next[i] = -1;
        if (idx[i] < 0 || !experts[i]) continue;
        int j = 0; for (; j < nu; j++) if (uniq[j] == idx[i]) break;
        if (j == nu) { uniq[nu] = idx[i]; head[nu] = i; cnt[nu] = 1; nu++; }
        else { int t = head[j]; while (next[t] >= 0) t = next[t]; next[t] = i; cnt[j]++; }
    }
    if (nu == 0) { memset(out, 0, (size_t)S * H * sizeof(float)); return; }

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    int cF = (4 * T + nu - 1) / nu; if (cF < 1) cF = 1; if (cF > F / 8) cF = F / 8 > 0 ? F / 8 : 1;
    int cH = (4 * T + nu - 1) / nu; if (cH < 1) cH = 1; if (cH > H / 8) cH = H / 8 > 0 ? H / 8 : 1;
    int rowsF = (F + cF - 1) / cF; rowsF = (rowsF + 7) & ~7; cF = (F + rowsF - 1) / rowsF;
    int rowsH = (H + cH - 1) / cH; rowsH = (rowsH + 7) & ~7; cH = (H + rowsH - 1) / rowsH;

    /* per-32 activation sums once per token (the K-quant min terms) */
    const int xsH = (H % 32) ? 0 : H / 32, xsF = (F % 32) ? 0 : F / 32;
    if (xsH) for (int s = 0; s < S; s++) gq_xsum32(x + (size_t)s * H, H, xs + (size_t)s * xsH);
    /* phase 1: gate+up rows [r0,r1) of every (expert, chunk), for every token routed there */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cF; it++) {
        int e = it / cF, c = it % cF;
        int r0 = c * rowsF, r1 = r0 + rowsF; if (r1 > F) r1 = F;
        for (int i = head[e]; i >= 0; i = next[i]) {
            const GqExpert *ex = experts[i]; const int s = i / K; const float *xi = x + (size_t)s * H;
            const float *xsi = xsH ? xs + (size_t)s * xsH : NULL;
            const size_t rg = gq_row_bytes(ex->tg, H), ru = gq_row_bytes(ex->tu, H);
            float *gi = g + (size_t)i * F, *ui = u + (size_t)i * F;
            for (int r = r0; r < r1; r++) {
                gi[r] = gq_dot_row_xs(ex->tg, ex->g + rg * (size_t)r, xi, xsi, H);
                ui[r] = gq_dot_row_xs(ex->tu, ex->u + ru * (size_t)r, xi, xsi, H);
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n; i++) {
        if (idx[i] < 0 || !experts[i]) continue;
        gq_swiglu(h + (size_t)i * F, g + (size_t)i * F, u + (size_t)i * F, F);
        if (xsF) gq_xsum32(h + (size_t)i * F, F, hs + (size_t)i * xsF);
    }
    /* phase 2: down rows [r0,r1) of every (expert, chunk) */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cH; it++) {
        int e = it / cH, c = it % cH;
        int r0 = c * rowsH, r1 = r0 + rowsH; if (r1 > H) r1 = H;
        for (int i = head[e]; i >= 0; i = next[i]) {
            const GqExpert *ex = experts[i]; const size_t rd = gq_row_bytes(ex->td, F);
            const float *hi = h + (size_t)i * F, *hsi = xsF ? hs + (size_t)i * xsF : NULL; float *ci = ctb + (size_t)i * H;
            for (int r = r0; r < r1; r++) ci[r] = gq_dot_row_xs(ex->td, ex->d + rd * (size_t)r, hi, hsi, F);
        }
    }
    /* rank-order sum per token */
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H;
        memset(os, 0, (size_t)H * sizeof(float));
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0 || !experts[i]) continue;
            float wgt = val[i]; const float *c = ctb + (size_t)i * H;
            for (int d = 0; d < H; d++) os[d] += wgt * c[d];
        }
    }
}

/* ---- self-test (startup): returns the number of failed checks ------------------------- */
static inline int gq_selftest(void) {
    int bad = 0;
    /* Q8_0 block: d = 1.0 (0x3C00), qs = -16..15 -> y[j] = j - 16 */
    uint8_t b8[34]; gq_st16(b8, 0x3C00); for (int j = 0; j < 32; j++) b8[2 + j] = (uint8_t)(int8_t)(j - 16);
    float y[256]; gq_deq_block_q8_0(b8, y);
    for (int j = 0; j < 32; j++) bad += y[j] != (float)(j - 16);
    /* Q6_K block: d = 0.5, all scales 2, every q = 0 -> every value = 0.5*2*(0-32) = -32 */
    uint8_t b6[210]; memset(b6, 0, sizeof b6); gq_st16(b6 + 208, 0x3800); for (int k = 0; k < 16; k++) b6[192 + k] = 2;
    gq_deq_block_q6_k(b6, y); for (int j = 0; j < 256; j++) bad += y[j] != -32.f;
    /* Q4_K block: d = 1, dmin = 1, scale/min pairs (3, 1) everywhere, q = 0xF -> 3*15 - 1 = 44 */
    uint8_t b4[144]; memset(b4, 0xFF, sizeof b4); gq_st16(b4, 0x3C00); gq_st16(b4 + 2, 0x3C00);
    for (int j = 0; j < 4; j++) { b4[4 + j] = 3; b4[8 + j] = 1; b4[12 + j] = 0x13; }
    gq_deq_block_q4_k(b4, y); for (int j = 0; j < 256; j++) bad += y[j] != 44.f;
    /* dot == sum of dequant for a one-hot activation */
    float x[32] = {0}; x[5] = 2.f; bad += gq_dot_q8_0_ref(b8, x, 32) != 2.f * (5 - 16);
    bad += gq_dot_row(GQ_Q8_0, b8, x, 32) != 2.f * (5 - 16);
    /* split round trip */
    int8_t plane[32]; float sc[1]; uint8_t back[34];
    gq_q8_0_split(b8, 32, 1, plane, sc); gq_q8_0_join(plane, sc, 32, 1, back);
    bad += memcmp(b8, back, 34) != 0;
    bad += gq_f16_to_f32(0x0001) != gq_bits_f32(0x33800000u);      /* smallest subnormal */
    bad += gq_f32_to_f16(gq_f16_to_f32(0xFBFF)) != 0xFBFF;         /* -65504 round trip */
    return bad;
}

#endif /* COLI_GQ_H */
