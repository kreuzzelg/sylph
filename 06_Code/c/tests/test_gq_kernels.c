/* gq.h: the ggml block kernels behind GGUF support. Contract and cases:
 * 07_Tests/IntegrationTest/gq_kernels.md (sylph).
 *
 *   (no args)                       unit suite, "all passed" + exit 0
 *   deq <TYPE> <blocks.bin> <numel> scalar-reference dequant -> LE f32 on stdout
 *                                   (exit 2 "unsupported type", exit 3 "size mismatch")
 *   moe-digest                      FNV-1a of one fixed gq_moe_run scenario
 *
 * The suite pins: the type table against gguf.h; the E0 golden vectors in
 * tests/fixtures/gq_e0/ (llama.cpp's gguf-py decode, bit for bit); every SIMD
 * dot equal to its scalar reference bit for bit; the reference within 1e-6 of a
 * double dot over the exact dequantized row; the Q8_0 split lossless and
 * consumable by gsgemv.h's matmul_q_gs; gq_matmul / gq_embed_row against the
 * row kernels; gq_moe_run against the per-token loop it replaces. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif
#include "../compat.h"
#include "../gq.h"
#include "../gguf.h"
#include "../gsgemv.h"

static unsigned g_seed = 12345;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static float frand(void) { return ((int)(rnd() & 0xFFFF) - 32768) / 8192.f; }    /* [-4, 4) */
static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static const int TYPES[8] = { GQ_F32, GQ_F16, GQ_BF16, GQ_Q4_0, GQ_Q8_0, GQ_Q4_K, GQ_Q5_K, GQ_Q6_K };

/* finite f16: wide = any exponent 0..30, else a small realistic weight scale */
static uint16_t rand_f16(int wide) {
    if (wide) return (uint16_t)(((rnd() & 1) << 15) | ((rnd() % 31) << 10) | (rnd() & 0x3FF));
    float v = (0.0005f + (rnd() & 1023) / 20000.f) * ((rnd() & 1) ? 1.f : -1.f);   /* |v| in [5e-4, 0.052) */
    return gq_f32_to_f16(v);
}
/* nb blocks of `type` with random payload and finite scale fields */
static void fill_blocks(int type, int nb, uint8_t *dst, int wide) {
    int ts = gq_types[type].tsize;
    for (int b = 0; b < nb; b++) {
        uint8_t *p = dst + (size_t)b * ts;
        for (int i = 0; i < ts; i++) p[i] = (uint8_t)rnd();
        switch (type) {
        case GQ_F32:  { float v = wide ? frand() * 1000.f : frand() * 0.01f; memcpy(p, &v, 4); break; }
        case GQ_F16:  gq_st16(p, rand_f16(wide)); break;
        case GQ_BF16: { float v = wide ? frand() * 1000.f : frand() * 0.01f; gq_st16(p, (uint16_t)(gq_f32_bits(v) >> 16)); break; }
        case GQ_Q4_0: case GQ_Q8_0: gq_st16(p, rand_f16(wide)); break;
        case GQ_Q4_K: case GQ_Q5_K: gq_st16(p, rand_f16(wide)); gq_st16(p + 2, rand_f16(wide)); break;
        case GQ_Q6_K: gq_st16(p + 208, rand_f16(wide)); break;
        }
    }
}
static uint8_t *rand_rows(int type, int O, int I, int wide) {
    size_t rb = gq_row_bytes(type, I);
    uint8_t *w = malloc(rb * O);
    fill_blocks(type, (int)(rb * O / gq_types[type].tsize), w, wide);
    return w;
}

/* ---- golden fixtures ---------------------------------------------------------------- */
static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz > 0 ? (size_t)sz : 1);
    *n = fread(b, 1, (size_t)sz, f); fclose(f); return b;
}
/* fixtures live next to the binary (tests/fixtures/gq_e0/); also accept the
 * engine directory or tests/ as cwd */
static const char *g_exe_dir = "";
static uint8_t *read_fixture(const char *stem, const char *ext, size_t *n) {
    char path[1024];
    snprintf(path, sizeof path, "%sfixtures/gq_e0/%s.%s", g_exe_dir, stem, ext);
    uint8_t *b = read_file(path, n); if (b) return b;
    snprintf(path, sizeof path, "tests/fixtures/gq_e0/%s.%s", stem, ext);
    b = read_file(path, n); if (b) return b;
    snprintf(path, sizeof path, "fixtures/gq_e0/%s.%s", stem, ext);
    return read_file(path, n);
}
static void test_golden(void) {
    for (int t = 0; t < 8; t++) {
        char stem[32]; snprintf(stem, sizeof stem, "synth_%s", gq_types[TYPES[t]].name);
        for (char *c = stem; *c; c++) if (*c >= 'A' && *c <= 'Z') *c = (char)(*c - 'A' + 'a');
        size_t nraw, nexp; uint8_t *raw = read_fixture(stem, "bin", &nraw); uint8_t *exp = read_fixture(stem, "f32", &nexp);
        if (!raw || !exp) { CHECK(0, "golden fixture tests/fixtures/gq_e0/%s.{bin,f32} missing", stem); free(raw); free(exp); continue; }
        int numel = (int)(nexp / 4);
        CHECK(gq_row_bytes(TYPES[t], numel) == nraw, "%s: raw size %zu != row bytes for %d elements", stem, nraw, numel);
        float *y = malloc(nexp);
        CHECK(gq_deq_row(TYPES[t], raw, y, numel) == 0, "%s: deq refused", stem);
        int nd = 0; for (int i = 0; i < numel; i++) if (memcmp(y + i, exp + 4 * (size_t)i, 4)) nd++;
        CHECK(nd == 0, "%s: %d of %d dequantized values differ from llama.cpp's gguf-py golden", stem, nd, numel);
        free(raw); free(exp); free(y);
    }
}

/* ---- dots: SIMD == scalar bit for bit, scalar within 1e-6 of double ------------------ */
static void test_dots(int type, int I, int O, int wide) {
    uint8_t *w = rand_rows(type, O, I, wide); size_t rb = gq_row_bytes(type, I);
    float *x = malloc(sizeof(float) * I); for (int i = 0; i < I; i++) x[i] = frand();
    float *deq = malloc(sizeof(float) * I);
    float *xs = malloc(sizeof(float) * (I / 32 + 1)); if (I % 32 == 0) gq_xsum32(x, I, xs);
    double worst = 0;
    for (int o = 0; o < O; o++) {
        const uint8_t *row = w + rb * (size_t)o;
        float r = gq_dot_row_ref(type, row, x, I), k = gq_dot_row(type, row, x, I);
        if (memcmp(&r, &k, 4)) { CHECK(0, "%s I=%d row %d: kernel %.9g != reference %.9g", gq_types[type].name, I, o, k, r); break; }
        if (I % 32 == 0) {
            float kx = gq_dot_row_xs(type, row, x, xs, I), rx = gq_dot_row_ref_xs(type, row, x, xs, I);
            if (memcmp(&r, &kx, 4) || memcmp(&r, &rx, 4)) { CHECK(0, "%s I=%d row %d: precomputed-xs path %.9g / %.9g != reference %.9g", gq_types[type].name, I, o, kx, rx, r); break; }
        }
        gq_deq_row(type, row, deq, I);
        double d = 0, mag = 0; for (int i = 0; i < I; i++) { double t = (double)deq[i] * x[i]; d += t; mag += fabs(t); }
        double e = mag > 0 ? fabs(d - r) / mag : fabs(d - r); if (e > worst) worst = e;
    }
    CHECK(worst < 1e-6, "%s I=%d: reference vs double dot err %.3g of the summed magnitude", gq_types[type].name, I, worst);
    free(w); free(x); free(deq); free(xs);
}

/* ---- Q8_0 split -> gsgemv.h ----------------------------------------------------------- */
static void test_split(int I, int O) {
    uint8_t *w = rand_rows(GQ_Q8_0, O, I, 1); size_t rb = gq_row_bytes(GQ_Q8_0, I);
    int8_t *plane = malloc((size_t)I * O); float *sc = malloc(sizeof(float) * (I / 32) * O); uint8_t *back = malloc(rb * O);
    gq_q8_0_split(w, I, O, plane, sc); gq_q8_0_join(plane, sc, I, O, back);
    CHECK(memcmp(w, back, rb * O) == 0, "Q8_0 split/join I=%d O=%d is not lossless", I, O);
    float *x = malloc(sizeof(float) * I); for (int i = 0; i < I; i++) x[i] = frand();
    float *y = malloc(sizeof(float) * O); float *deq = malloc(sizeof(float) * I);
    matmul_q_gs(y, x, plane, sc, I, O, 32);
    double worst = 0, worst_k = 0;
    for (int o = 0; o < O; o++) {
        gq_deq_row(GQ_Q8_0, w + rb * (size_t)o, deq, I);
        double d = 0, mag = 0; for (int i = 0; i < I; i++) { double t = (double)deq[i] * x[i]; d += t; mag += fabs(t); }
        double e = fabs(d - y[o]) / mag; if (e > worst) worst = e;
        double ek = fabs((double)gq_dot_row(GQ_Q8_0, w + rb * (size_t)o, x, I) - y[o]) / mag; if (ek > worst_k) worst_k = ek;
    }
    CHECK(worst < 1e-6, "matmul_q_gs on the split vs double: err %.3g (I=%d)", worst, I);
    CHECK(worst_k < 1e-6, "matmul_q_gs on the split vs gq_dot_q8_0: err %.3g (I=%d)", worst_k, I);
    free(w); free(plane); free(sc); free(back); free(x); free(y); free(deq);
}

/* ---- dense ------------------------------------------------------------------------------ */
static void test_dense(int type, int I, int O) {
    uint8_t *w = rand_rows(type, O, I, 1); size_t rb = gq_row_bytes(type, I);
    float *x = malloc(sizeof(float) * I); for (int i = 0; i < I; i++) x[i] = frand();
    float *y = malloc(sizeof(float) * O);
    CHECK(gq_matmul(y, x, type, w, I, O) == 0, "gq_matmul %s refused", gq_types[type].name);
    int bad = 0; for (int o = 0; o < O; o++) { float r = gq_dot_row(type, w + rb * (size_t)o, x, I); bad += memcmp(&r, y + o, 4) != 0; }
    CHECK(bad == 0, "gq_matmul %s: %d rows differ from gq_dot_row", gq_types[type].name, bad);
    float *a = malloc(sizeof(float) * I), *b = malloc(sizeof(float) * I);
    int row = O - 1; gq_embed_row(type, w, row, a, I); gq_deq_row(type, w + rb * (size_t)row, b, I);
    CHECK(memcmp(a, b, sizeof(float) * I) == 0, "gq_embed_row %s != gq_deq_row", gq_types[type].name);
    CHECK(gq_matmul(y, x, GGML_TYPE_Q2_K, w, I, O) == -1 && gq_embed_row(GGML_TYPE_IQ4_NL, w, 0, a, I) == -1, "unsupported types not refused by the dense entry points");
    free(w); free(x); free(y); free(a); free(b);
}

/* ---- layer runner ------------------------------------------------------------------------- */
typedef struct { uint8_t *g, *u, *d; GqExpert e; } Ex;
static void make_expert(Ex *ex, int H, int F, int td) {
    ex->g = rand_rows(GQ_Q4_K, F, H, 0); ex->u = rand_rows(GQ_Q4_K, F, H, 0); ex->d = rand_rows(td, H, F, 0);
    ex->e.g = ex->g; ex->e.u = ex->u; ex->e.d = ex->d; ex->e.tg = GQ_Q4_K; ex->e.tu = GQ_Q4_K; ex->e.td = td;
}
static void free_expert(Ex *ex) { free(ex->g); free(ex->u); free(ex->d); }
/* the per-token loop gq_moe_run replaces */
static void moe_loop(float *out, const float *x, int S, int K, int H, int F, const int *idx, const float *val, const GqExpert *const *ex) {
    float *g = malloc(sizeof(float) * F), *u = malloc(sizeof(float) * F), *h = malloc(sizeof(float) * F), *c = malloc(sizeof(float) * H);
    for (int s = 0; s < S; s++) {
        float *os = out + (size_t)s * H; memset(os, 0, sizeof(float) * H);
        for (int k = 0; k < K; k++) {
            int i = s * K + k; if (idx[i] < 0 || !ex[i]) continue;
            const GqExpert *e = ex[i];
            for (int r = 0; r < F; r++) { g[r] = gq_dot_row(e->tg, e->g + gq_row_bytes(e->tg, H) * (size_t)r, x + (size_t)s * H, H); u[r] = gq_dot_row(e->tu, e->u + gq_row_bytes(e->tu, H) * (size_t)r, x + (size_t)s * H, H); }
            for (int r = 0; r < F; r++) h[r] = (g[r] / (1.f + expf(-g[r]))) * u[r];
            for (int o = 0; o < H; o++) c[o] = gq_dot_row(e->td, e->d + gq_row_bytes(e->td, F) * (size_t)o, h, F);
            for (int o = 0; o < H; o++) os[o] += val[i] * c[o];
        }
    }
    free(g); free(u); free(h); free(c);
}
static void setup_layer(int S, int K, int E, int *idx, float *val, const GqExpert **ex, Ex *pool) {
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) {
        int i = s * K + k; int e = (int)(rnd() % E); for (int j = 0; j < k; j++) if (idx[s * K + j] == e) e = (e + 1) % E;
        idx[i] = e; val[i] = 0.05f + (rnd() & 255) / 512.f; ex[i] = &pool[e].e;
    }
}
static void test_layer(int S, int K, int H, int F, int E) {
    Ex *pool = malloc(sizeof(Ex) * E); for (int e = 0; e < E; e++) make_expert(&pool[e], H, F, e & 1 ? GQ_Q6_K : GQ_Q5_K);
    int *idx = malloc(sizeof(int) * S * K); float *val = malloc(sizeof(float) * S * K); const GqExpert **ex = malloc(sizeof(GqExpert *) * S * K);
    setup_layer(S, K, E, idx, val, ex, pool);
    float *x = malloc(sizeof(float) * S * H); for (int i = 0; i < S * H; i++) x[i] = frand();
    float *out = malloc(sizeof(float) * S * H), *ref = malloc(sizeof(float) * S * H);
    void *scratch = malloc(gq_moe_scratch_bytes(S, K, H, F));
    gq_moe_run(out, x, S, K, H, F, idx, val, ex, scratch);
    moe_loop(ref, x, S, K, H, F, idx, val, ex);
    int bad = 0; float worst = 0;
    for (int i = 0; i < S * H; i++) { float diff = fabsf(out[i] - ref[i]); if (diff > 1e-5f + 1e-4f * fabsf(ref[i])) bad++; if (diff > worst) worst = diff; }
    CHECK(bad == 0, "gq_moe_run S=%d K=%d H=%d F=%d E=%d: %d outputs differ from the per-token loop (worst |diff| %.3g)", S, K, H, F, E, bad, worst);
    free(scratch); free(out); free(ref); free(x); free(idx); free(val); free(ex);
    for (int e = 0; e < E; e++) free_expert(&pool[e]);
    free(pool);
}

/* ---- CLI modes --------------------------------------------------------------------------- */
static int type_by_name(const char *name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) if (gguf_types[t].name && !strcmp(gguf_types[t].name, name)) return t;
    return -1;
}
static int cmd_deq(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: deq <TYPE> <blocks.bin> <numel>\n"); return 1; }
    int t = type_by_name(argv[2]);
    if (!gq_supported(t)) { fprintf(stderr, "unsupported type %s\n", argv[2]); return 2; }
    long long numel = atoll(argv[4]);
    size_t n; uint8_t *raw = read_file(argv[3], &n);
    if (!raw) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
    if (numel <= 0 || numel > (1 << 30) || gq_row_bytes(t, (int)numel) != n) {
        fprintf(stderr, "size mismatch: %zu bytes is not %lld %s elements (%zu expected)\n", n, numel, argv[2], gq_row_bytes(t, (int)numel)); return 3;
    }
    float *y = malloc(sizeof(float) * (size_t)numel);
    gq_deq_row(t, raw, y, (int)numel);
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    fwrite(y, sizeof(float), (size_t)numel, stdout);
    free(y); free(raw); return 0;
}
static int cmd_moe_digest(void) {
    const int S = 7, K = 8, H = 2048, F = 512, E = 12;
    g_seed = 20261005;
    Ex *pool = malloc(sizeof(Ex) * E); for (int e = 0; e < E; e++) make_expert(&pool[e], H, F, e & 1 ? GQ_Q6_K : GQ_Q5_K);
    int idx[7 * 8]; float val[7 * 8]; const GqExpert *ex[7 * 8];
    setup_layer(S, K, E, idx, val, ex, pool);
    float *x = malloc(sizeof(float) * S * H); for (int i = 0; i < S * H; i++) x[i] = frand();
    float *out = malloc(sizeof(float) * S * H); void *scratch = malloc(gq_moe_scratch_bytes(S, K, H, F));
    gq_moe_run(out, x, S, K, H, F, idx, val, ex, scratch);
    uint64_t hsh = 0xcbf29ce484222325ull; const uint8_t *b = (const uint8_t *)out;
    for (size_t i = 0; i < sizeof(float) * S * H; i++) { hsh ^= b[i]; hsh *= 0x100000001b3ull; }
    printf("%016llx\n", (unsigned long long)hsh);
    free(scratch); free(out); free(x); for (int e = 0; e < E; e++) free_expert(&pool[e]); free(pool);
    return 0;
}

/* The int8-activation twin (gq_i8.h, FR-14): a deviation, so it is pinned by
 * TOLERANCE against the f32 reference and the deltas are printed, not hidden.
 * Int8 activations carry up to d/2 = amax/254 of error per element; the dot's
 * error is bounded by that times the row's L1 norm, so the bound is relative
 * to sum |w_i| * max |x|, not to the (possibly cancelling) result. The AVX2
 * twin sums the same integers in another order: 1e-5 of the same scale. */
static void test_i8(int type, int I, int O) {
    uint8_t *w = rand_rows(type, O, I, 0); size_t rb = gq_row_bytes(type, I);
    float *x = malloc((size_t)I * sizeof(float)), *deq = malloc((size_t)I * sizeof(float));
    void *buf = malloc(gq_act8_bytes(I)); GqAct8 a; gq_act8_bind(&a, buf, I);
    for (int i = 0; i < I; i++) x[i] = frand();
    gq_act8_quantize(x, I, &a);
    float amax = 0.f; for (int i = 0; i < I; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
    double worst = 0, worst_simd = 0;
    for (int o = 0; o < O; o++) {
        const uint8_t *row = w + rb * (size_t)o;
        float ref = gq_dot_row_ref(type, row, x, I), i8 = gq_dot8_row_ref(type, row, &a, I), fast = gq_dot8_row(type, row, &a, I);
        gq_deq_row(type, row, deq, I); double l1 = 0; for (int i = 0; i < I; i++) l1 += fabs((double)deq[i]);
        double scale = l1 * amax / 254.0 + 1e-6;                      /* the per-element bound times the L1 norm */
        double e = fabs((double)i8 - (double)ref) / scale, es = fabs((double)fast - (double)i8) / (l1 * amax + 1e-6);
        if (e > worst) worst = e; if (es > worst_simd) worst_simd = es;
    }
    printf("i8 twin %s I=%d: |i8 - f32| <= %.3f of the error bound, SIMD vs scalar %.2e relative\n", gq_type_name(type), I, worst, worst_simd);
    CHECK(worst <= 1.0, "%s I=%d: int8 twin exceeds the activation error bound (%.3f x)", gq_type_name(type), I, worst);
    CHECK(worst_simd <= 1e-5, "%s I=%d: AVX2 int8 twin differs from the scalar twin by %.2e", gq_type_name(type), I, worst_simd);
    free(w); free(x); free(deq); free(buf);
}
static void test_layer_i8(int S, int K, int H, int F, int E) {
    uint8_t **g = malloc(sizeof *g * E), **u = malloc(sizeof *u * E), **d = malloc(sizeof *d * E);
    GqExpert *ex = malloc(sizeof *ex * E); const GqExpert **sel = malloc(sizeof *sel * (size_t)S * K);
    int *idx = malloc(sizeof(int) * (size_t)S * K); float *val = malloc(sizeof(float) * (size_t)S * K);
    for (int e = 0; e < E; e++) { int tg = (e & 1) ? GQ_Q4_K : GQ_Q5_K, td = (e & 2) ? GQ_Q6_K : GQ_Q8_0;
        g[e] = rand_rows(tg, F, H, 0); u[e] = rand_rows(tg, F, H, 0); d[e] = rand_rows(td, H, F, 0);
        ex[e].g = g[e]; ex[e].u = u[e]; ex[e].d = d[e]; ex[e].tg = ex[e].tu = tg; ex[e].td = td; }
    for (int i = 0; i < S * K; i++) { idx[i] = (int)(rnd() % E); val[i] = 0.1f + (rnd() & 255) / 512.f; sel[i] = &ex[idx[i]]; }
    float *x = malloc((size_t)S * H * sizeof(float)), *o32 = malloc((size_t)S * H * sizeof(float)), *o8 = malloc((size_t)S * H * sizeof(float));
    for (int i = 0; i < S * H; i++) x[i] = frand();
    void *sc = malloc(gq_moe_scratch_bytes(S, K, H, F));
    gq_moe_run(o32, x, S, K, H, F, idx, val, sel, sc);
    gq_moe_run_i8(o8, x, S, K, H, F, idx, val, sel, sc);
    double mx = 0, dm = 0; for (int i = 0; i < S * H; i++) { if (fabs(o32[i]) > mx) mx = fabs(o32[i]); double dd = fabs((double)o32[i] - o8[i]); if (dd > dm) dm = dd; }
    printf("i8 twin layer S=%d K=%d H=%d F=%d: max |delta| %.3e vs max |out| %.3e (%.2f%%)\n", S, K, H, F, dm, mx, 100.0 * dm / (mx + 1e-30));
    CHECK(dm <= 0.05 * mx + 1e-4, "int8 layer twin deviates by %.3e (max |out| %.3e)", dm, mx);
    for (int e = 0; e < E; e++) { free(g[e]); free(u[e]); free(d[e]); }
    free(g); free(u); free(d); free(ex); free(sel); free(idx); free(val); free(x); free(o32); free(o8); free(sc);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "deq")) return cmd_deq(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "moe-digest")) return cmd_moe_digest();
    static char exe_dir[1024];
    { const char *slash = strrchr(argv[0], '/');
#ifdef _WIN32
      const char *bs = strrchr(argv[0], '\\'); if (!slash || (bs && bs > slash)) slash = bs;
#endif
      if (slash && (size_t)(slash - argv[0]) < sizeof exe_dir - 2) { memcpy(exe_dir, argv[0], (size_t)(slash - argv[0]) + 1); g_exe_dir = exe_dir; } }
    printf("path: %s\n", GQ_PATH);

    /* type table against gguf.h; exactly the v1 set */
    int nsup = 0;
    for (int t = 0; t < GQ_TYPE_COUNT; t++) {
        if (!gq_supported(t)) continue;
        nsup++;
        CHECK(gguf_type_known(t) && gq_types[t].block == gguf_types[t].block && gq_types[t].tsize == gguf_types[t].tsize, "type %d geometry differs from gguf.h", t);
        CHECK(gq_type_id(gq_types[t].name) == t && !strcmp(gq_type_name(t), gguf_type_name(t)), "type %d name round trip", t);
        CHECK((int64_t)gq_row_bytes(t, 256) == gguf_row_size(t, 256), "row bytes for 256 elements of %s", gq_type_name(t));
    }
    CHECK(nsup == 8, "%d supported types, expected 8", nsup);
    CHECK(gq_row_bytes(GQ_Q4_K, 100) == 0 && gq_row_bytes(GGML_TYPE_Q2_K, 256) == 0 && gq_type_id("Q2_K") == -1, "non-multiples / unsupported types must yield 0 / -1");
    CHECK(gq_selftest() == 0, "gq_selftest reports %d failures", gq_selftest());
    /* conversions: every f16 bit pattern round-trips through f32 (finite ones through f32_to_f16 too) */
    { int bad = 0; for (unsigned h = 0; h < 65536; h++) { float f = gq_f16_to_f32((uint16_t)h); if (((h >> 10) & 0x1F) != 31 && gq_f32_to_f16(f) != h) bad++; }
      CHECK(bad == 0, "%d finite f16 values do not round-trip", bad); }

    test_golden();

    const int Is[4] = { 256, 512, 2048, 4096 };
    for (int t = 0; t < 8; t++) for (int i = 0; i < 4; i++) { test_dots(TYPES[t], Is[i], 6, 1); test_dots(TYPES[t], Is[i], 2, 0); }
    for (int t = 0; t < 3; t++) test_dots(TYPES[t], 40, 3, 1);        /* non-multiple of 32: scalar fallback */

    test_split(2048, 8); test_split(512, 5); test_split(64, 1);
    test_dense(GQ_Q6_K, 2048, 300); test_dense(GQ_Q8_0, 2048, 300); test_dense(GQ_Q4_K, 2048, 300);

    /* an unreachable routing (all -1) must zero the output */
    { int idx[4] = {-1, -1, -1, -1}; float val[4] = {0}; const GqExpert *ex[4] = {0}; float x[256], out[256];
      void *sc = malloc(gq_moe_scratch_bytes(1, 4, 256, 256));
      for (int i = 0; i < 256; i++) { x[i] = 1.f; out[i] = 7.f; }
      gq_moe_run(out, x, 1, 4, 256, 256, idx, val, ex, sc); int nz = 0; for (int i = 0; i < 256; i++) nz += out[i] != 0.f;
      CHECK(nz == 0, "unrouted token left %d non-zero outputs", nz); free(sc); }
    test_layer(1, 8, 2048, 512, 16);
    test_layer(7, 8, 2048, 512, 12);
    test_layer(5, 4, 256, 256, 6);

    /* the int8-activation twin (FR-14, opt-in): tolerance-pinned, deltas printed */
    { const int T8[4] = { GQ_Q8_0, GQ_Q4_K, GQ_Q5_K, GQ_Q6_K };
      for (int t = 0; t < 4; t++) { test_i8(T8[t], 2048, 64); test_i8(T8[t], 512, 64); } }
    test_layer_i8(1, 8, 2048, 512, 16);
    test_layer_i8(5, 4, 256, 256, 6);

    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    printf("all passed\n"); return 0;
}
