/* Kernel oracle for the raw ggml block formats on the device (sylph phase 5).
 * Contract: 07_Tests/IntegrationTest/cuda_tier_kquant.md, part B. Runs on the
 * owner's card through `make cuda-test-gq`; the CPU reference is gq.h compiled
 * as C into tests/gq_ref.o (gq_ref.c), so the oracle is the engine's own code.
 *
 *   B1  dense GEMV on fmt 24/28/29/30 tensors, S in {1,3}: every output
 *       bit-identical to gq_dot_row_ref; row s of S=3 equals S=1; two launches
 *       identical.
 *   B2  expert group issue/take (K in {1,2,8}, mixed types): the GEMV halves are
 *       B1 on the same tensors; the fused output is within the layer tolerance
 *       1e-5 + 1e-4*|ref| of the host reference (device expf in the SiLU is the
 *       one named deviation); the maximum |delta| is printed.
 *   B3  refusals: I not a whole number of blocks, a scale array, fmt 18 (Q4_0).
 *   B4  determinism under load: B1 fifty times with other work in flight.
 *
 * Shapes are the Qwen3.6-35B-A3B ones (hidden 2048, expert inter 512) plus an
 * lm_head-like 2048 x 4096 for the dense path. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "../backend_gpu_compat.h"
#include <cuda_runtime.h>

extern "C" {
void gq_ref_seed(unsigned s);
int gq_ref_supported(int type);
int gq_ref_block_elems(int type);
int gq_ref_block_bytes(int type);
size_t gq_ref_row_bytes(int type, int I);
const char *gq_ref_type_name(int type);
float gq_ref_dot(int type, const unsigned char *row, const float *x, int I);
float gq_ref_dot_fast(int type, const unsigned char *row, const float *x, int I);
int gq_ref_deq_row(int type, const unsigned char *row, float *out, int I);
void gq_ref_fill_blocks(int type, int nb, unsigned char *dst, int wide);
void gq_ref_fill_x(float *x, int n);
}
#include "../backend_cuda.h"

enum { T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_Q4_0 = 2 };
static const int TYPES[4] = { T_Q8_0, T_Q4_K, T_Q5_K, T_Q6_K };
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static int fmt_of(int type) { return COLI_FMT_GGML_BASE + type; }

static unsigned char *rows(int type, int O, int I, int wide) {
    size_t rb = gq_ref_row_bytes(type, I);
    unsigned char *w = (unsigned char *)malloc(rb * O);
    gq_ref_fill_blocks(type, (int)(rb * O / gq_ref_block_bytes(type)), w, wide);
    return w;
}
static int bits_equal(float a, float b) { unsigned ua, ub; memcpy(&ua, &a, 4); memcpy(&ub, &b, 4); return ua == ub; }

/* --- B1: dense GEMV bit-identical to the reference --------------------------- */
static void dense_case(int type, int I, int O, int wide) {
    size_t rb = gq_ref_row_bytes(type, I);
    unsigned char *w = rows(type, O, I, wide);
    ColiCudaTensor *t = NULL;
    if (!coli_cuda_tensor_upload(&t, w, NULL, fmt_of(type), I, O, 0)) { CHECK(0, "upload %s %dx%d", gq_ref_type_name(type), O, I); free(w); return; }
    std::vector<float> x(3 * (size_t)I), y1((size_t)O), y3(3 * (size_t)O), y1b((size_t)O);
    gq_ref_fill_x(x.data(), 3 * I);
    CHECK(coli_cuda_matmul(&t, y1.data(), x.data(), NULL, NULL, fmt_of(type), 1, I, O, 0, 0), "matmul S=1 %s", gq_ref_type_name(type));
    CHECK(coli_cuda_matmul(&t, y3.data(), x.data(), NULL, NULL, fmt_of(type), 3, I, O, 0, 0), "matmul S=3 %s", gq_ref_type_name(type));
    CHECK(coli_cuda_matmul(&t, y1b.data(), x.data(), NULL, NULL, fmt_of(type), 1, I, O, 0, 0), "matmul S=1 again");
    int bad_ref = 0, bad_s = 0, bad_rep = 0, bad_fast = 0; int first = -1;
    for (int o = 0; o < O; o++) {
        float r = gq_ref_dot(type, w + rb * (size_t)o, x.data(), I);
        if (!bits_equal(r, y1[o])) { if (first < 0) first = o; bad_ref++; }
        if (!bits_equal(y1[o], y3[o])) bad_s++;
        if (!bits_equal(y1[o], y1b[o])) bad_rep++;
        if (!bits_equal(r, gq_ref_dot_fast(type, w + rb * (size_t)o, x.data(), I))) bad_fast++;
        for (int s = 1; s < 3; s++) {
            float rs = gq_ref_dot(type, w + rb * (size_t)o, x.data() + (size_t)s * I, I);
            if (!bits_equal(rs, y3[(size_t)s * O + o])) bad_s++;
        }
    }
    if (bad_ref) printf("    %s %dx%d: first mismatch row %d: device %.9g reference %.9g\n", gq_ref_type_name(type), O, I, first, y1[first], gq_ref_dot(type, w + rb * (size_t)first, x.data(), I));
    CHECK(bad_ref == 0, "%s %dx%d (%s scales): %d of %d rows differ from gq_dot_row_ref", gq_ref_type_name(type), O, I, wide ? "wide" : "narrow", bad_ref, O);
    CHECK(bad_s == 0, "%s %dx%d: S=3 rows differ from S=1 / reference (%d)", gq_ref_type_name(type), O, I, bad_s);
    CHECK(bad_rep == 0, "%s %dx%d: two launches differ (%d)", gq_ref_type_name(type), O, I, bad_rep);
    CHECK(bad_fast == 0, "%s: the host SIMD path differs from the scalar reference (%d) -- gq_kernels.md regression, not a device fault", gq_ref_type_name(type), bad_fast);
    coli_cuda_tensor_free(t); free(w);
}

/* --- B2: fused expert vs host reference --------------------------------------- */
static void expert_case(int K, int tg, int td, int D, int Ih) {
    std::vector<ColiCudaTensor *> g(K), u(K), d(K);
    std::vector<unsigned char *> wg(K), wu(K), wd(K);
    std::vector<int> rws(K, 1);
    size_t rbg = gq_ref_row_bytes(tg, D), rbd = gq_ref_row_bytes(td, Ih);
    for (int k = 0; k < K; k++) {
        wg[k] = rows(tg, Ih, D, 0); wu[k] = rows(tg, Ih, D, 0); wd[k] = rows(td, D, Ih, 0);
        g[k] = u[k] = d[k] = NULL;
        CHECK(coli_cuda_tensor_upload(&g[k], wg[k], NULL, fmt_of(tg), D, Ih, 0), "upload gate %d", k);
        CHECK(coli_cuda_tensor_upload(&u[k], wu[k], NULL, fmt_of(tg), D, Ih, 0), "upload up %d", k);
        CHECK(coli_cuda_tensor_upload(&d[k], wd[k], NULL, fmt_of(td), Ih, D, 0), "upload down %d", k);
    }
    std::vector<float> x((size_t)K * D); gq_ref_fill_x(x.data(), K * D);
    CHECK(coli_cuda_expert_group_issue(g.data(), u.data(), d.data(), rws.data(), K, x.data()), "group issue K=%d", K);
    const float *y = coli_cuda_expert_group_take(0);
    CHECK(y != NULL, "group take K=%d", K);
    if (!y) return;
    double worst = 0; int bad = 0;
    std::vector<float> gg(Ih), uu(Ih), hh(Ih), ref(D);
    for (int k = 0; k < K; k++) {
        const float *xk = x.data() + (size_t)k * D;
        for (int i = 0; i < Ih; i++) {
            gg[i] = gq_ref_dot(tg, wg[k] + rbg * (size_t)i, xk, D);
            uu[i] = gq_ref_dot(tg, wu[k] + rbg * (size_t)i, xk, D);
            float gv = gg[i]; hh[i] = (gv / (1.f + expf(-gv))) * uu[i];
        }
        for (int o = 0; o < D; o++) {
            ref[o] = gq_ref_dot(td, wd[k] + rbd * (size_t)o, hh.data(), Ih);
            double delta = fabs((double)ref[o] - (double)y[(size_t)k * D + o]);
            if (delta > worst) worst = delta;
            if (delta > 1e-5 + 1e-4 * fabs((double)ref[o])) bad++;
        }
    }
    printf("    expert K=%d %s/%s: max |delta| vs host reference %.3g (device expf in the SiLU)\n", K, gq_ref_type_name(tg), gq_ref_type_name(td), worst);
    CHECK(bad == 0, "expert K=%d: %d outputs outside 1e-5 + 1e-4*|ref|", K, bad);
    for (int k = 0; k < K; k++) { coli_cuda_tensor_free(g[k]); coli_cuda_tensor_free(u[k]); coli_cuda_tensor_free(d[k]); free(wg[k]); free(wu[k]); free(wd[k]); }
}

int main() {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { puts("no CUDA device: cuda-test-gq skipped"); return 0; }
    gq_ref_seed(20261009);
    size_t cnt0 = 0, bytes0 = 0;

    puts("B1 dense GEMV == gq_dot_row_ref");
    const int shapes[3][2] = { {2048, 512}, {512, 2048}, {2048, 4096} };
    for (int ti = 0; ti < 4; ti++) for (int si = 0; si < 3; si++) for (int wide = 0; wide < 2; wide++)
        dense_case(TYPES[ti], shapes[si][0], shapes[si][1], wide);

    puts("B2 fused expert vs host reference");
    expert_case(1, T_Q4_K, T_Q6_K, 2048, 512);
    expert_case(2, T_Q4_K, T_Q6_K, 2048, 512);
    expert_case(8, T_Q4_K, T_Q5_K, 2048, 512);
    expert_case(8, T_Q8_0, T_Q8_0, 2048, 512);

    puts("B3 refusals");
    coli_cuda_stats(0, &cnt0, &bytes0);
    {
        ColiCudaTensor *t = NULL; unsigned char junk[34 * 4] = {0}; float sc[4] = {1, 1, 1, 1};
        CHECK(!coli_cuda_tensor_upload(&t, junk, NULL, fmt_of(T_Q8_0), 100, 1, 0), "Q8_0 with I=100 must be refused");
        CHECK(!coli_cuda_tensor_upload(&t, junk, NULL, fmt_of(T_Q4_K), 200, 1, 0), "Q4_K with I=200 must be refused");
        CHECK(!coli_cuda_tensor_upload(&t, junk, sc, fmt_of(T_Q8_0), 32, 1, 0), "a block tensor with a scale array must be refused");
        CHECK(!coli_cuda_tensor_upload(&t, junk, NULL, fmt_of(T_Q4_0), 32, 1, 0), "fmt 18 (Q4_0) is not a device format in v1");
        size_t cnt1 = 0, bytes1 = 0; coli_cuda_stats(0, &cnt1, &bytes1);
        CHECK(cnt1 == cnt0 && bytes1 == bytes0, "refusals allocate nothing (%zu/%zu -> %zu/%zu)", cnt0, bytes0, cnt1, bytes1);
    }

    puts("B4 determinism under load");
    {
        int type = T_Q4_K, I = 2048, O = 512; size_t rb = gq_ref_row_bytes(type, I);
        unsigned char *w = rows(type, O, I, 0); ColiCudaTensor *t = NULL, *busy = NULL;
        unsigned char *wb = rows(T_Q6_K, 4096, 2048, 0);
        CHECK(coli_cuda_tensor_upload(&t, w, NULL, fmt_of(type), I, O, 0), "upload");
        CHECK(coli_cuda_tensor_upload(&busy, wb, NULL, fmt_of(T_Q6_K), 2048, 4096, 0), "upload busy");
        std::vector<float> x(I), y(O), yb(4096), ref(O); gq_ref_fill_x(x.data(), I);
        for (int o = 0; o < O; o++) ref[o] = gq_ref_dot(type, w + rb * (size_t)o, x.data(), I);
        int bad = 0;
        for (int r = 0; r < 50; r++) {
            coli_cuda_matmul(&busy, yb.data(), x.data(), NULL, NULL, fmt_of(T_Q6_K), 1, 2048, 4096, 0, 0);
            coli_cuda_matmul(&t, y.data(), x.data(), NULL, NULL, fmt_of(type), 1, I, O, 0, 0);
            for (int o = 0; o < O; o++) if (!bits_equal(y[o], ref[o])) bad++;
        }
        CHECK(bad == 0, "under load: %d mismatching outputs over 50 launches", bad);
        coli_cuda_tensor_free(t); coli_cuda_tensor_free(busy); free(w); free(wb);
    }

    coli_cuda_shutdown();
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    puts("cuda-test-gq: all passed");
    return 0;
}
