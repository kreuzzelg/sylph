/* bench_gq: kernel throughput of gq.h against upstream's planar-int4 and int8
 * kernels at Qwen3.6-35B-A3B shapes. Protocol and pass criteria:
 * 07_Tests/SystemTest/kernel_throughput.md (NFR-7). Prints one Markdown table.
 *
 *   make bench-gq [MODEL=/path/to/model.gguf]      # MODEL: expert 0 of blk.0 on real bytes
 *
 * Rows: A gate GEMV Q4_K vs planar int4 · B down GEMV Q5_K/Q6_K vs planar ·
 * C dense 8192x2048 Q8_0 (native, split+matmul_q_gs) vs int8-row · D lm_head
 * 248320x2048 (Q6_K, Q8_0, split) vs int8-row, ms/token · E whole MoE layer
 * gq_moe_run vs xf_moe_run at S=1 and S=32, hot and cold · F embed row.
 * Hot = same bytes every iteration; cold = rotate through 8 copies (> L3).
 * Median of 50 timed iterations after 5 warm-ups. Rows A/B/F single-threaded
 * (kernel speed per byte); C/D/E use the OpenMP team. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../gq.h"
#include "../expert_ffn.h"
#include "../gsgemv.h"
#include "../gguf.h"

static unsigned g_seed = 777;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static float frand(void) { return ((int)(rnd() & 0xFFFF) - 32768) / 8192.f; }

static double now(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
#define WARM 5
#define ITER 50
/* median seconds of one call of body (a statement) */
#define MEDIAN(out, body) do { double _t[ITER]; for (int _i = 0; _i < WARM + ITER; _i++) { double _a = now(); body; double _b = now(); if (_i >= WARM) _t[_i - WARM] = _b - _a; } qsort(_t, ITER, sizeof(double), cmp_d); out = _t[ITER / 2]; } while (0)

static volatile float g_sink;   /* keeps the dot results alive */

/* ---- weight generators (realistic scales) ------------------------------------------- */
static uint16_t small_f16(void) { float v = (0.0005f + (rnd() & 1023) / 20000.f) * ((rnd() & 1) ? 1.f : -1.f); return gq_f32_to_f16(v); }
static void fill_blocks(int type, size_t nb, uint8_t *dst) {
    int ts = gq_types[type].tsize;
    for (size_t b = 0; b < nb; b++) {
        uint8_t *p = dst + b * ts;
        for (int i = 0; i < ts; i++) p[i] = (uint8_t)rnd();
        if (type == GQ_Q8_0 || type == GQ_Q4_0) gq_st16(p, small_f16());
        else if (type == GQ_Q4_K || type == GQ_Q5_K) { gq_st16(p, small_f16()); gq_st16(p + 2, small_f16()); }
        else if (type == GQ_Q6_K) gq_st16(p + 208, small_f16());
    }
}
static uint8_t *rows(int type, int O, int I) { size_t rb = gq_row_bytes(type, I); uint8_t *w = malloc(rb * O); fill_blocks(type, rb * O / gq_types[type].tsize, w); return w; }
/* planar int4 gs64 rows + scales (upstream's expert layout) */
static void planar(int O, int I, uint8_t **w, float **sc) {
    *w = malloc((size_t)O * I / 2); *sc = malloc(sizeof(float) * O * (I / 64));
    for (size_t i = 0; i < (size_t)O * I / 2; i++) (*w)[i] = (uint8_t)rnd();
    for (size_t i = 0; i < (size_t)O * (I / 64); i++) (*sc)[i] = 0.0005f + (rnd() & 1023) / 20000.f;
}

typedef struct { const char *row, *kernel, *layout; double bytes_row, us, gbs, ratio; const char *note; } Line;
static Line lines[64]; static int nlines;
static void add(const char *row, const char *kernel, const char *layout, double bytes_row, double us_row, double ratio_ns_per_byte, const char *note) {
    Line *L = &lines[nlines++]; L->row = row; L->kernel = kernel; L->layout = layout; L->bytes_row = bytes_row; L->us = us_row;
    L->gbs = bytes_row / (us_row * 1e3); L->ratio = ratio_ns_per_byte; L->note = note;
}

/* ---- rows A/B: single-thread row dots ------------------------------------------------- */
static double bench_rows_gq(int type, const uint8_t *w, int O, int I, const float *x) {
    size_t rb = gq_row_bytes(type, I); double t;
    float *xs = malloc(sizeof(float) * (I / 32));
    MEDIAN(t, { float s = 0; gq_xsum32(x, I, xs); for (int o = 0; o < O; o++) s += gq_dot_row_xs(type, w + rb * (size_t)o, x, xs, I); g_sink = s; });
    free(xs); return t / O * 1e6;
}
static double bench_rows_planar(const uint8_t *w, const float *sc, int O, int I, const float *x) {
    double t; size_t rb = (size_t)I / 2; int ng = I / 64;
    MEDIAN(t, { float s = 0; for (int o = 0; o < O; o++) s += xf_dot_f32(w + rb * (size_t)o, sc + (size_t)o * ng, x, I); g_sink = s; });
    return t / O * 1e6;
}
static void row_ab(const char *tag, int type, const uint8_t *real, int O, int I) {
    float *x = malloc(sizeof(float) * I); for (int i = 0; i < I; i++) x[i] = frand();
    uint8_t *pw; float *ps; planar(O, I, &pw, &ps);
    double base = bench_rows_planar(pw, ps, O, I, x), bbytes = I / 2.0 + 4.0 * (I / 64);
    double base_nsb = base * 1e3 / bbytes;
    add(tag, "xf_dot_f32 (baseline)", "planar int4 gs64", bbytes, base, 1.0, "upstream expert kernel");
    uint8_t *w = real ? NULL : rows(type, O, I);
    double us = bench_rows_gq(type, real ? real : w, O, I, x), bytes = (double)gq_row_bytes(type, I);
    char *k = malloc(48); snprintf(k, 48, "gq_dot_row %s", gq_type_name(type));
    add(tag, k, real ? "real GGUF rows" : "synthetic blocks", bytes, us, (us * 1e3 / bytes) / base_nsb, "ratio = ns per weight byte vs baseline");
    free(x); free(pw); free(ps); free(w);
}

/* ---- rows C/D: dense GEMV with the OpenMP team ------------------------------------------ */
static void row_dense(const char *tag, int I, int O, int with_q6, const char *unit_note) {
    float *x = malloc(sizeof(float) * I); for (int i = 0; i < I; i++) x[i] = frand();
    float *y = malloc(sizeof(float) * O);
    uint8_t *q8 = rows(GQ_Q8_0, O, I);
    int8_t *plane = malloc((size_t)I * O); float *sc32 = malloc(sizeof(float) * (size_t)O * (I / 32)); float *scrow = malloc(sizeof(float) * O);
    gq_q8_0_split(q8, I, O, plane, sc32);
    for (int o = 0; o < O; o++) scrow[o] = sc32[(size_t)o * (I / 32)];
    double t_row, t_split, t_native, t_q6 = 0;
    MEDIAN(t_row, matmul_q_gs(y, x, plane, scrow, I, O, I));
    MEDIAN(t_split, matmul_q_gs(y, x, plane, sc32, I, O, 32));
    MEDIAN(t_native, gq_matmul(y, x, GQ_Q8_0, q8, I, O));
    double b_row = (double)I + 4.0, b_split = (double)I + 4.0 * (I / 32), b_q8 = (double)gq_row_bytes(GQ_Q8_0, I);
    double base_nsb = t_row * 1e9 / ((double)O * b_row);
    add(tag, "matmul_q_gs gs=I (baseline)", "int8-row + 1 scale", b_row, t_row / O * 1e6, 1.0, unit_note);
    add(tag, "matmul_q_gs gs=32", "Q8_0 split (plane + f32/32)", b_split, t_split / O * 1e6, (t_split * 1e9 / ((double)O * b_split)) / base_nsb, "gq_q8_0_split at load, upstream kernel");
    add(tag, "gq_matmul Q8_0", "raw Q8_0 blocks", b_q8, t_native / O * 1e6, (t_native * 1e9 / ((double)O * b_q8)) / base_nsb, "native block kernel");
    if (with_q6) {
        uint8_t *q6 = rows(GQ_Q6_K, O, I);
        MEDIAN(t_q6, gq_matmul(y, x, GQ_Q6_K, q6, I, O));
        double b_q6 = (double)gq_row_bytes(GQ_Q6_K, I);
        add(tag, "gq_matmul Q6_K", "raw Q6_K blocks", b_q6, t_q6 / O * 1e6, (t_q6 * 1e9 / ((double)O * b_q6)) / base_nsb, "unsloth output.weight type");
        free(q6);
    }
    if (O >= 100000) printf("lm_head ms/token: int8-row %.2f · Q8_0 split %.2f · Q8_0 native %.2f%s%.2f\n", t_row * 1e3, t_split * 1e3, t_native * 1e3, with_q6 ? " · Q6_K " : "", with_q6 ? t_q6 * 1e3 : 0.0);
    free(x); free(y); free(q8); free(plane); free(sc32); free(scrow);
}

/* ---- row E: whole layer, hot and cold ------------------------------------------------------ */
#define COPIES 8
typedef struct { uint8_t *g, *u, *d; GqExpert e; } GqEx;
typedef struct { uint8_t *g4, *u4, *d4; float *gs, *us, *ds; XfExpert e; } XfEx;
static void row_layer(int S, int K, int H, int F, int E, const uint8_t *rg, const uint8_t *ru, const uint8_t *rd, int rdtype) {
    int *idx = malloc(sizeof(int) * S * K); float *val = malloc(sizeof(float) * S * K);
    for (int s = 0; s < S; s++) for (int k = 0; k < K; k++) { int e = (int)(rnd() % E); for (int j = 0; j < k; j++) if (idx[s * K + j] == e) e = (e + 1) % E; idx[s * K + k] = e; val[s * K + k] = 0.05f + (rnd() & 255) / 512.f; }
    float *x = malloc(sizeof(float) * S * H); for (int i = 0; i < S * H; i++) x[i] = frand();
    float *out = malloc(sizeof(float) * S * H);
    GqEx *gq = malloc(sizeof(GqEx) * E * COPIES); XfEx *xf = malloc(sizeof(XfEx) * E * COPIES);
    const GqExpert **gex = malloc(sizeof(void *) * S * K * COPIES); const XfExpert **xex = malloc(sizeof(void *) * S * K * COPIES);
    for (int c = 0; c < COPIES; c++) for (int e = 0; e < E; e++) {
        GqEx *q = &gq[c * E + e]; XfEx *p = &xf[c * E + e];
        q->g = rows(GQ_Q4_K, F, H); q->u = rows(GQ_Q4_K, F, H); q->d = rows(rdtype, H, F);
        if (rg && c == 0 && e == 0) { memcpy(q->g, rg, gq_row_bytes(GQ_Q4_K, H) * F); memcpy(q->u, ru, gq_row_bytes(GQ_Q4_K, H) * F); memcpy(q->d, rd, gq_row_bytes(rdtype, F) * H); }
        q->e.g = q->g; q->e.u = q->u; q->e.d = q->d; q->e.tg = GQ_Q4_K; q->e.tu = GQ_Q4_K; q->e.td = rdtype;
        planar(F, H, &p->g4, &p->gs); planar(F, H, &p->u4, &p->us); planar(H, F, &p->d4, &p->ds);
        p->e.g4 = p->g4; p->e.u4 = p->u4; p->e.d4 = p->d4; p->e.gs = p->gs; p->e.us = p->us; p->e.ds = p->ds;
    }
    for (int c = 0; c < COPIES; c++) for (int i = 0; i < S * K; i++) { gex[c * S * K + i] = &gq[c * E + idx[i]].e; xex[c * S * K + i] = &xf[c * E + idx[i]].e; }
    void *sgq = malloc(gq_moe_scratch_bytes(S, K, H, F)), *sxf = malloc(xf_moe_scratch_bytes(S, K, H, F));
    double t_xf_hot, t_gq_hot, t_xf_cold, t_gq_cold; int c = 0;
    MEDIAN(t_xf_hot, xf_moe_run(out, x, S, K, H, F, idx, val, xex, 0, sxf));
    MEDIAN(t_gq_hot, gq_moe_run(out, x, S, K, H, F, idx, val, gex, sgq));
    MEDIAN(t_xf_cold, { xf_moe_run(out, x, S, K, H, F, idx, val, xex + (size_t)c * S * K, 0, sxf); c = (c + 1) % COPIES; });
    MEDIAN(t_gq_cold, { gq_moe_run(out, x, S, K, H, F, idx, val, gex + (size_t)c * S * K, sgq); c = (c + 1) % COPIES; });
    double bx = (double)E * (2.0 * F * (H / 2.0 + 4.0 * (H / 64)) + H * (F / 2.0 + 4.0 * (F / 64)));
    double bq = (double)E * (2.0 * F * gq_row_bytes(GQ_Q4_K, H) + (double)H * gq_row_bytes(rdtype, F));
    char *tag = malloc(16); snprintf(tag, 16, "E S=%d", S);
    char *kq = malloc(64); snprintf(kq, 64, "gq_moe_run Q4_K/Q4_K/%s", gq_type_name(rdtype));
    add(tag, "xf_moe_run hot (baseline)", "planar int4 gs64", bx / E, t_xf_hot * 1e3, 1.0, "ms per layer, E resident experts");
    add(tag, kq, "raw K-quant blocks, hot", bq / E, t_gq_hot * 1e3, t_gq_hot / t_xf_hot, "ratio = layer time vs baseline");
    add(tag, "xf_moe_run cold", "planar int4 gs64", bx / E, t_xf_cold * 1e3, t_xf_cold / t_xf_hot, "fresh copy per iteration");
    add(tag, kq, "raw K-quant blocks, cold", bq / E, t_gq_cold * 1e3, t_gq_cold / t_xf_cold, "ratio = cold layer time vs cold baseline");
    for (int i = 0; i < E * COPIES; i++) { free(gq[i].g); free(gq[i].u); free(gq[i].d); free(xf[i].g4); free(xf[i].u4); free(xf[i].d4); free(xf[i].gs); free(xf[i].us); free(xf[i].ds); }
    free(gq); free(xf); free(gex); free(xex); free(sgq); free(sxf); free(idx); free(val); free(x); free(out);
}

/* ---- row F ---------------------------------------------------------------------------------- */
static void row_embed(int I) {
    uint8_t *q8 = rows(GQ_Q8_0, 64, I); float *f32 = malloc(sizeof(float) * I * 64); for (int i = 0; i < I * 64; i++) f32[i] = frand();
    float *out = malloc(sizeof(float) * I); double t_q8, t_cp;
    MEDIAN(t_cp, { for (int r = 0; r < 64; r++) memcpy(out, f32 + (size_t)r * I, sizeof(float) * I); g_sink = out[3]; });
    MEDIAN(t_q8, { for (int r = 0; r < 64; r++) gq_embed_row(GQ_Q8_0, q8, r, out, I); g_sink = out[3]; });
    add("F", "f32 row copy (baseline)", "f32", 4.0 * I, t_cp / 64 * 1e6, 1.0, "token_embd lookup");
    add("F", "gq_embed_row Q8_0", "raw Q8_0 blocks", (double)gq_row_bytes(GQ_Q8_0, I), t_q8 / 64 * 1e6, t_q8 / t_cp, "ratio = time per row vs copy");
    free(q8); free(f32); free(out);
}

/* ---- real expert slices from a GGUF (expert 0 of blk.0) --------------------------------------- */
static uint8_t *slice(GgufSet *G, const char *name, int want_type, int *type, int64_t *ne0, int64_t *ne1) {
    GgufTensor *t = gguf_find(G, name);
    if (!t) { fprintf(stderr, "bench_gq: %s not in the model\n", name); return NULL; }
    if (!gq_supported(t->type) || (want_type >= 0 && t->type != want_type && want_type != GQ_Q5_K)) { fprintf(stderr, "bench_gq: %s is %s\n", name, gguf_type_name(t->type)); return NULL; }
    *type = t->type; *ne0 = t->ne[0]; *ne1 = t->ne[1];
    size_t rb = gq_row_bytes(t->type, (int)t->ne[0]), n = rb * (size_t)t->ne[1];
    uint8_t *buf = malloc(n); int fd = G->files[t->file].fd;
    if (pread(fd, buf, n, t->off) != (ssize_t)n) { fprintf(stderr, "bench_gq: short read of %s\n", name); free(buf); return NULL; }
    return buf;
}

int main(int argc, char **argv) {
    const char *model = NULL;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    printf("# gq.h kernel throughput (NFR-7)\n\npath: %s · OpenMP threads: %d · compiler: %s · median of %d after %d warm-ups\n", GQ_PATH, T, __VERSION__, ITER, WARM);
    const int H = 2048, F = 512, K = 8;
    uint8_t *rg = NULL, *ru = NULL, *rd = NULL; int tg = GQ_Q4_K, tu = GQ_Q4_K, td = GQ_Q5_K;
    GgufSet G; memset(&G, 0, sizeof G);
    if (model) {
        if (gguf_open_set(&G, model, NULL) == 0) {
            int64_t a, b;
            rg = slice(&G, "blk.0.ffn_gate_exps.weight", GQ_Q4_K, &tg, &a, &b);
            ru = slice(&G, "blk.0.ffn_up_exps.weight", GQ_Q4_K, &tu, &a, &b);
            rd = slice(&G, "blk.0.ffn_down_exps.weight", GQ_Q5_K, &td, &a, &b);
            if (rg && ru && rd && (a != F || b != H || tg != GQ_Q4_K)) { fprintf(stderr, "bench_gq: unexpected expert shape/type, using synthetic\n"); free(rg); free(ru); free(rd); rg = ru = rd = NULL; td = GQ_Q5_K; }
            printf("model: %s (expert 0 of blk.0: gate/up %s, down %s)\n", model, gq_type_name(tg), gq_type_name(td));
        } else fprintf(stderr, "bench_gq: %s\n", G.err);
    }
    printf("\n");
    row_ab("A", GQ_Q4_K, rg, F, H);
    row_ab("B", GQ_Q5_K, td == GQ_Q5_K ? rd : NULL, H, F);
    row_ab("B", GQ_Q6_K, td == GQ_Q6_K ? rd : NULL, H, F);
    row_dense("C", H, 8192, 0, "attn_qkv shape 8192x2048, µs per row");
    row_dense("D", H, 248320, 1, "lm_head 248320x2048, µs per row");
    row_layer(1, K, H, F, 16, rg, ru, rd, td);
    row_layer(32, K, H, F, 16, rg, ru, rd, td);
    row_embed(H);

    printf("| row | kernel | layout | bytes/row | µs/row (E: ms/layer) | GB/s | ratio | note |\n|---|---|---|---|---|---|---|---|\n");
    for (int i = 0; i < nlines; i++) {
        Line *L = &lines[i];
        printf("| %s | %s | %s | %.0f | %.3f | %.2f | %.2f | %s |\n", L->row, L->kernel, L->layout, L->bytes_row, L->us, L->gbs, L->ratio, L->note);
    }
    printf("\nNFR-7: row A ratio ≤ 1.5 and row E (S=1, hot) ratio ≤ 1.5 pass the target; others are reported.\n");
    if (model) gguf_close(&G);
    free(rg); free(ru); free(rd);
    return 0;
}
