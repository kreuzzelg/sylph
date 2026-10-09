/* gq_i8.h: the int8-activation twin of gq.h's row dots (sylph, FR-14 / spec §4.2).
 *
 * OPT-IN and MEASURED, never the default: QWEN_EXPERT_ACT=i8 on a GGUF source
 * routes moe_gq_run through gq_moe_run_i8, every other setting keeps the f32
 * activations that E0-E3 are checked with. The deltas are recorded by the
 * equivalence harness (`--deviation QWEN_EXPERT_ACT=i8`, gpu_rtx3070.md G5).
 *
 * Activations are quantized per 32 elements to int8 with one f32 scale (the
 * block of ggml's Q8_0 activation; for the K-quants ggml uses one scale per
 * 256 -- Q8_K -- so this twin is at least as fine), plus the per-32 integer
 * sums the Q4_K/Q5_K min terms need. The dots are then integer: 32 products
 * into an int32 per sub-block, one float fma per sub-block (weight scale x
 * activation scale), the Q6_K per-16 scales likewise. Scalar reference for
 * every type; AVX2 (maddubs / madd) for the hot ones. The AVX2 path sums the
 * same integers, so it agrees with the scalar twin to the float rounding of
 * the per-block combination (checked within 1e-5 relative by
 * tests/test_gq_kernels), not bit for bit: this is a deviation already, not a
 * reference. Included by gq.h at its end. */
#ifndef COLI_GQ_I8_H
#define COLI_GQ_I8_H

typedef struct { int8_t *q; float *d; int32_t *bs; } GqAct8;   /* q[I], d[I/32], bs[I/32] = per-32 sums of q */

/* bytes of one quantized activation of length I (I % 32 == 0) */
static inline size_t gq_act8_bytes(int I) { return (size_t)I + (size_t)(I / 32) * (sizeof(float) + sizeof(int32_t)) + 64; }
static inline void gq_act8_bind(GqAct8 *a, void *buf, int I) {
    char *p = (char *)buf; a->q = (int8_t *)p; p += (size_t)I; p = (char *)(((uintptr_t)p + 15) & ~(uintptr_t)15);
    a->d = (float *)p; p += (size_t)(I / 32) * sizeof(float); a->bs = (int32_t *)p;
}
/* x -> per-32 int8 blocks: d = amax/127, q = roundf(x/d) (ggml's quantize_row_q8_0_ref rule) */
static inline void gq_act8_quantize(const float *x, int I, GqAct8 *a) {
    for (int g = 0; g < I / 32; g++) {
        const float *xb = x + 32 * g; float amax = 0.f;
        for (int i = 0; i < 32; i++) { float v = fabsf(xb[i]); if (v > amax) amax = v; }
        float d = amax / 127.f, id = d ? 1.f / d : 0.f; int32_t s = 0;
        for (int i = 0; i < 32; i++) { int q = (int)lrintf(xb[i] * id); if (q > 127) q = 127; if (q < -127) q = -127; a->q[32 * g + i] = (int8_t)q; s += q; }
        a->d[g] = d; a->bs[g] = s;
    }
}

/* ---- scalar reference ---------------------------------------------------------- */
static inline float gq_dot8_q8_0_ref(const uint8_t *w, const GqAct8 *a, int I) {
    float acc = 0.f;
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 34 * (size_t)g; const int8_t *q = a->q + 32 * g; int32_t s = 0;
        for (int i = 0; i < 32; i++) s += (int)(int8_t)b[2 + i] * (int)q[i];
        acc = fmaf((float)s, gq_f16_to_f32(gq_ld16(b)) * a->d[g], acc);
    }
    return acc;
}
static inline float gq_dot8_q45_k_ref(const uint8_t *w, const GqAct8 *a, int I, int five) {
    const size_t bs = five ? 176 : 144; float acc = 0.f;
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + bs * (size_t)g, *qh = b + 16, *ql = five ? b + 48 : b + 16;
        float d1[8], nm1[8]; gq_k4_scales(b + 4, gq_f16_to_f32(gq_ld16(b)), gq_f16_to_f32(gq_ld16(b + 2)), d1, nm1);
        for (int j = 0; j < 4; j++) for (int half = 0; half < 2; half++) {
            const int k = 2 * j + half, sb = 8 * g + k; const uint8_t u = (uint8_t)((half ? 2u : 1u) << (2 * j));
            const int8_t *q = a->q + 32 * sb; int32_t s = 0;
            for (int l = 0; l < 32; l++) { int v = (half ? (ql[32 * j + l] >> 4) : (ql[32 * j + l] & 0xF)) + (five && (qh[l] & u) ? 16 : 0); s += v * (int)q[l]; }
            acc = fmaf((float)s, d1[k] * a->d[sb], acc);
            acc = fmaf((float)a->bs[sb], nm1[k] * a->d[sb], acc);
        }
    }
    return acc;
}
static inline float gq_dot8_q6_k_ref(const uint8_t *w, const GqAct8 *a, int I) {
    float acc = 0.f;
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + 210 * (size_t)g; const float d = gq_f16_to_f32(gq_ld16(b + 208));
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = b + 64 * h, *qh = b + 128 + 32 * h; const int8_t *sc = (const int8_t *)(b + 192 + 8 * h);
            const int8_t *q = a->q + 256 * g + 128 * h;
            for (int k = 0; k < 8; k++) {
                int32_t s = 0;
                for (int t = 0; t < 16; t++) { int r = 16 * k + t; s += gq_q6_get(ql, qh, r) * (int)q[r]; }
                acc = fmaf((float)s, d * (float)sc[k] * a->d[(256 * g + 128 * h + 16 * k) / 32], acc);
            }
        }
    }
    return acc;
}
static inline float gq_dot8_row_ref(int type, const uint8_t *row, const GqAct8 *a, int I) {
    switch (type) {
    case GQ_Q8_0: return gq_dot8_q8_0_ref(row, a, I);
    case GQ_Q4_K: return gq_dot8_q45_k_ref(row, a, I, 0);
    case GQ_Q5_K: return gq_dot8_q45_k_ref(row, a, I, 1);
    case GQ_Q6_K: return gq_dot8_q6_k_ref(row, a, I);
    default: return 0.f;    /* f32/f16/bf16/Q4_0 rows: no int8 twin (see gq_dot8_row) */
    }
}
static inline int gq_dot8_supported(int type) { return type == GQ_Q8_0 || type == GQ_Q4_K || type == GQ_Q5_K || type == GQ_Q6_K; }

/* ---- AVX2 ---------------------------------------------------------------------- */
#ifdef GQ_HAVE_AVX2
/* sum of 8 int32 lanes */
static inline int32_t gq_i8_hsum32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8)); s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return _mm_cvtsi128_si32(s);
}
/* 32 unsigned weights (0..255 in bytes) x 32 signed int8 activations -> 8 int32 pair sums */
static inline __m256i gq_i8_dot_u8s8(__m256i wu, __m256i xs) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(wu, xs), _mm256_set1_epi16(1));
}
static inline float gq_dot8_q8_0_avx2(const uint8_t *w, const GqAct8 *a, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < I / 32; g++) {
        const uint8_t *b = w + 34 * (size_t)g;
        __m256i wv = _mm256_loadu_si256((const __m256i *)(b + 2)), xv = _mm256_loadu_si256((const __m256i *)(a->q + 32 * g));
        /* signed x signed: 16-bit widen both halves, madd */
        __m256i lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(wv)), _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv)));
        __m256i hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(wv, 1)), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(lo, hi)), _mm256_set1_ps(gq_f16_to_f32(gq_ld16(b)) * a->d[g]), acc);
    }
    return gq_hsum8(acc);
}
static inline float gq_dot8_q45_k_avx2(const uint8_t *w, const GqAct8 *a, int I, int five) {
    const size_t bs = five ? 176 : 144; __m256 acc = _mm256_setzero_ps(); float accm = 0.f;
    const __m256i m4 = _mm256_set1_epi8(0x0F), v16 = _mm256_set1_epi8(16);
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + bs * (size_t)g, *qh = b + 16, *ql = five ? b + 48 : b + 16;
        float d1[8], nm1[8]; gq_k4_scales(b + 4, gq_f16_to_f32(gq_ld16(b)), gq_f16_to_f32(gq_ld16(b + 2)), d1, nm1);
        __m256i hv = five ? _mm256_loadu_si256((const __m256i *)qh) : _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {
            __m256i qv = _mm256_loadu_si256((const __m256i *)(ql + 32 * j));
            __m256i lo = _mm256_and_si256(qv, m4), hi = _mm256_and_si256(_mm256_srli_epi16(qv, 4), m4);
            if (five) {
                __m256i blo = _mm256_set1_epi8((char)(1u << (2 * j))), bhi = _mm256_set1_epi8((char)(2u << (2 * j)));
                lo = _mm256_add_epi8(lo, _mm256_and_si256(v16, _mm256_cmpeq_epi8(_mm256_and_si256(hv, blo), blo)));
                hi = _mm256_add_epi8(hi, _mm256_and_si256(v16, _mm256_cmpeq_epi8(_mm256_and_si256(hv, bhi), bhi)));
            }
            for (int half = 0; half < 2; half++) {
                const int k = 2 * j + half, sb = 8 * g + k;
                __m256i xv = _mm256_loadu_si256((const __m256i *)(a->q + 32 * sb));
                __m256i s = gq_i8_dot_u8s8(half ? hi : lo, xv);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d1[k] * a->d[sb]), acc);
                accm = fmaf((float)a->bs[sb], nm1[k] * a->d[sb], accm);
            }
        }
    }
    return gq_hsum8(acc) + accm;
}
static inline float gq_dot8_q6_k_avx2(const uint8_t *w, const GqAct8 *a, int I) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(3), c32 = _mm256_set1_epi8(32);
    for (int g = 0; g < I / GQ_QK_K; g++) {
        const uint8_t *b = w + 210 * (size_t)g; const float d = gq_f16_to_f32(gq_ld16(b + 208));
        for (int h = 0; h < 2; h++) {
            const uint8_t *ql = b + 64 * h, *qh = b + 128 + 32 * h; const int8_t *sc = (const int8_t *)(b + 192 + 8 * h);
            const int8_t *q = a->q + 256 * g + 128 * h; const float *dx = a->d + (256 * g + 128 * h) / 32;
            __m256i l0 = _mm256_loadu_si256((const __m256i *)ql), l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i hv = _mm256_loadu_si256((const __m256i *)qh);
            /* the four 32-element parts of gq_q6_get, as (q + 32) in 0..63 */
            __m256i part[4];
            part[0] = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(hv, m3), 4));
            part[1] = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 2), m3), 4));
            part[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 4), m3), 4));
            part[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hv, 6), m3), 4));
            for (int p = 0; p < 4; p++) {
                /* part p covers elements 32p..32p+31 = sub-blocks k = 2p (lanes 0-3) and 2p+1 (lanes 4-7) */
                __m256i xv = _mm256_loadu_si256((const __m256i *)(q + 32 * p));
                __m256i s = gq_i8_dot_u8s8(part[p], xv);                                   /* sum (q+32) * x */
                __m256i corr = gq_i8_dot_u8s8(c32, xv);                                   /* 32 * sum x, same lanes */
                __m256i si = _mm256_sub_epi32(s, corr);
                const float sA = d * (float)sc[2 * p] * dx[p], sB = d * (float)sc[2 * p + 1] * dx[p];
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(si), _mm256_setr_ps(sA, sA, sA, sA, sB, sB, sB, sB), acc);
            }
        }
    }
    return gq_hsum8(acc);
}
static inline float gq_dot8_row(int type, const uint8_t *row, const GqAct8 *a, int I) {
    if (I % 32) return gq_dot8_row_ref(type, row, a, I);
    switch (type) {
    case GQ_Q8_0: return gq_dot8_q8_0_avx2(row, a, I);
    case GQ_Q4_K: return gq_dot8_q45_k_avx2(row, a, I, 0);
    case GQ_Q5_K: return gq_dot8_q45_k_avx2(row, a, I, 1);
    case GQ_Q6_K: return gq_dot8_q6_k_avx2(row, a, I);
    default: return 0.f;
    }
}
#else
static inline float gq_dot8_row(int type, const uint8_t *row, const GqAct8 *a, int I) { return gq_dot8_row_ref(type, row, a, I); }
#endif

/* ---- the layer runner's int8 twin -------------------------------------------------
 * Same contract as gq_moe_run (rank-ordered sum, thread-count independent); the
 * activations x (per token) and h (per token x expert) are quantized once and
 * every row dot is integer. A type without an int8 twin (f32/f16/bf16/Q4_0 rows)
 * falls back to the f32 dot for that matrix. The scratch comes from
 * gq_moe_scratch_bytes, which reserves room for both modes. */
static inline size_t gq_moe_i8_extra_bytes(int S, int K, int H, int F) {
    return (size_t)S * gq_act8_bytes(H) + (size_t)S * K * gq_act8_bytes(F) + 128;
}
static inline float gq_dot8_or_f32(int type, const uint8_t *row, const GqAct8 *a, const float *x, const float *xs, int I) {
    return gq_dot8_supported(type) && I % 32 == 0 ? gq_dot8_row(type, row, a, I) : gq_dot_row_xs(type, row, x, xs, I);
}
static inline void gq_moe_run_i8(float *out, const float *x, int S, int K, int H, int F,
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
    char *ax   = GQ_TAKE(char, (size_t)S * gq_act8_bytes(H));
    char *ah   = GQ_TAKE(char, n * gq_act8_bytes(F));
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
    const int xsH = (H % 32) ? 0 : H / 32, xsF = (F % 32) ? 0 : F / 32;
    /* activations once per token: int8 blocks (and the f32 lane sums for a non-twin type) */
    for (int s = 0; s < S; s++) {
        if (xsH) { gq_xsum32(x + (size_t)s * H, H, xs + (size_t)s * xsH); GqAct8 a; gq_act8_bind(&a, ax + (size_t)s * gq_act8_bytes(H), H); gq_act8_quantize(x + (size_t)s * H, H, &a); }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cF; it++) {
        int e = it / cF, c = it % cF;
        int r0 = c * rowsF, r1 = r0 + rowsF; if (r1 > F) r1 = F;
        for (int i = head[e]; i >= 0; i = next[i]) {
            const GqExpert *ex = experts[i]; const int s = i / K; const float *xi = x + (size_t)s * H;
            const float *xsi = xsH ? xs + (size_t)s * xsH : NULL;
            GqAct8 a; gq_act8_bind(&a, ax + (size_t)s * gq_act8_bytes(H), H);
            const size_t rg = gq_row_bytes(ex->tg, H), ru = gq_row_bytes(ex->tu, H);
            float *gi = g + (size_t)i * F, *ui = u + (size_t)i * F;
            for (int r = r0; r < r1; r++) {
                gi[r] = gq_dot8_or_f32(ex->tg, ex->g + rg * (size_t)r, &a, xi, xsi, H);
                ui[r] = gq_dot8_or_f32(ex->tu, ex->u + ru * (size_t)r, &a, xi, xsi, H);
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)n; i++) {
        if (idx[i] < 0 || !experts[i]) continue;
        gq_swiglu(h + (size_t)i * F, g + (size_t)i * F, u + (size_t)i * F, F);
        if (xsF) { gq_xsum32(h + (size_t)i * F, F, hs + (size_t)i * xsF); GqAct8 a; gq_act8_bind(&a, ah + (size_t)i * gq_act8_bytes(F), F); gq_act8_quantize(h + (size_t)i * F, F, &a); }
    }
    #pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < nu * cH; it++) {
        int e = it / cH, c = it % cH;
        int r0 = c * rowsH, r1 = r0 + rowsH; if (r1 > H) r1 = H;
        for (int i = head[e]; i >= 0; i = next[i]) {
            const GqExpert *ex = experts[i]; const size_t rd = gq_row_bytes(ex->td, F);
            const float *hi = h + (size_t)i * F, *hsi = xsF ? hs + (size_t)i * xsF : NULL; float *ci = ctb + (size_t)i * H;
            GqAct8 a; gq_act8_bind(&a, ah + (size_t)i * gq_act8_bytes(F), F);
            for (int r = r0; r < r1; r++) ci[r] = gq_dot8_or_f32(ex->td, ex->d + rd * (size_t)r, &a, hi, hsi, F);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H;
        memset(os, 0, (size_t)H * sizeof(float));
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0 || !experts[i]) continue;
            float wgt = val[i]; const float *c = ctb + (size_t)i * H;
            for (int d = 0; d < H; d++) os[d] = GQ_FMA(wgt, c[d], os[d]);
        }
    }
}

#endif /* COLI_GQ_I8_H */
