/* Device throughput of the raw-block kernels (sylph phase 8, architecture amendment
 * 2026-10-17). `make bench-gq-cuda [BENCH_ITERS=n]` on a card: for every trunk shape of
 * Qwen3.6-35B-A3B and for an expert group, the median time of the kernel the engine
 * runs and the bandwidth it achieved (weight bytes + activation bytes per launch).
 *
 *   dense   coli_cuda_tensor_gemm_async (device pointers: the chain's and the DeltaNet
 *           step's entry; the tier's dense path launches the same blk_matmul)
 *   group   coli_cuda_expert_group_issue/take (K experts, host pointers: includes the
 *           x upload and the y download the tier pays per token)
 *
 * The oracle (test_gq_cuda.cu) says the numbers are right; this says how fast. Both
 * read gq.h through tests/gq_ref.o, so the shapes and fixtures are the engine's. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../backend_gpu_compat.h"
#include <cuda_runtime.h>

extern "C" {
void gq_ref_seed(unsigned s);
int gq_ref_block_bytes(int type);
size_t gq_ref_row_bytes(int type, int I);
const char *gq_ref_type_name(int type);
void gq_ref_fill_blocks(int type, int nb, unsigned char *dst, int wide);
void gq_ref_fill_x(float *x, int n);
}
#include "../backend_cuda.h"
extern "C" int coli_cuda_tensor_gemm_async(ColiCudaTensor *t, float *y_dev, const float *x_dev, int S, void *stream);

enum { T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14 };
static int fmt_of(int type) { return COLI_FMT_GGML_BASE + type; }
static unsigned char *rows(int type, int O, int I) {
    size_t rb = gq_ref_row_bytes(type, I);
    unsigned char *w = (unsigned char *)malloc(rb * O);
    gq_ref_fill_blocks(type, (int)(rb * O / gq_ref_block_bytes(type)), w, 0);
    return w;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static int iters() { const char *e = getenv("BENCH_ITERS"); int n = e ? atoi(e) : 50; return n < 3 ? 3 : n; }

/* dense: y = W x on the device, timed with events around the launch (no transfers) */
static void dense(const char *what, int type, int I, int O) {
    size_t rb = gq_ref_row_bytes(type, I);
    unsigned char *w = rows(type, O, I); ColiCudaTensor *t = NULL;
    if (!coli_cuda_tensor_upload(&t, w, NULL, fmt_of(type), I, O, 0)) { printf("  %-34s upload refused\n", what); free(w); return; }
    std::vector<float> x((size_t)I); gq_ref_fill_x(x.data(), I);
    float *xd = NULL, *yd = NULL; cudaMalloc(&xd, (size_t)I * sizeof(float)); cudaMalloc(&yd, (size_t)O * sizeof(float));
    cudaMemcpy(xd, x.data(), (size_t)I * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; i++) coli_cuda_tensor_gemm_async(t, yd, xd, 1, NULL);
    cudaDeviceSynchronize();
    /* ten back-to-back launches per sample (as a decode frame issues them: the card stays
     * at its working clock, launch gaps amortise), the median over the samples */
    std::vector<double> us;
    for (int i = 0; i < iters(); i++) {
        cudaEventRecord(a, 0);
        for (int k = 0; k < 10; k++) coli_cuda_tensor_gemm_async(t, yd, xd, 1, NULL);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b); us.push_back(ms * 1e3 / 10.0);
    }
    double m = median(us), mn = *std::min_element(us.begin(), us.end()), bytes = (double)rb * O + (double)I * 4;
    printf("  %-34s %-5s %7d x %-5d %7.1f MB  %8.1f us  %6.1f GB/s  (best %6.1f GB/s)\n", what, gq_ref_type_name(type), O, I, bytes / 1e6, m, bytes / m / 1e3, bytes / mn / 1e3);
    cudaFree(xd); cudaFree(yd); coli_cuda_tensor_free(t); free(w);
}

/* group: K experts of gate/up [Ih x D] type tg and down [D x Ih] type td, one token, the tier's
 * issue/take round trip (x up, h2d of the descriptors, two kernels, y down) timed on the host */
static void group(const char *what, int K, int tg, int td, int D, int Ih) {
    std::vector<ColiCudaTensor *> g(K, (ColiCudaTensor *)NULL), u(K, (ColiCudaTensor *)NULL), d(K, (ColiCudaTensor *)NULL);
    std::vector<unsigned char *> wg(K), wu(K), wd(K); std::vector<int> rws(K, 1);
    for (int k = 0; k < K; k++) {
        wg[k] = rows(tg, Ih, D); wu[k] = rows(tg, Ih, D); wd[k] = rows(td, D, Ih);
        if (!coli_cuda_tensor_upload(&g[k], wg[k], NULL, fmt_of(tg), D, Ih, 0) || !coli_cuda_tensor_upload(&u[k], wu[k], NULL, fmt_of(tg), D, Ih, 0) ||
            !coli_cuda_tensor_upload(&d[k], wd[k], NULL, fmt_of(td), Ih, D, 0)) { printf("  %-34s upload refused\n", what); return; }
    }
    std::vector<float> x((size_t)K * D); gq_ref_fill_x(x.data(), K * D);
    for (int i = 0; i < 5; i++) { if (coli_cuda_expert_group_issue(g.data(), u.data(), d.data(), rws.data(), K, x.data())) coli_cuda_expert_group_take(0); }
    std::vector<double> us;
    for (int i = 0; i < iters(); i++) {
        double t0 = now_ms();
        if (!coli_cuda_expert_group_issue(g.data(), u.data(), d.data(), rws.data(), K, x.data())) { printf("  %-34s issue refused\n", what); return; }
        const float *y = coli_cuda_expert_group_take(0); (void)y;
        us.push_back((now_ms() - t0) * 1e3);
    }
    double m = median(us), bytes = (double)K * (2.0 * gq_ref_row_bytes(tg, D) * Ih + (double)gq_ref_row_bytes(td, Ih) * D) + (double)K * D * 8;
    printf("  %-34s %s/%s K=%d %4dx%-4d %7.1f MB  %8.1f us  %6.1f GB/s  (issue+take, host clock)\n", what, gq_ref_type_name(tg), gq_ref_type_name(td), K, Ih, D, bytes / 1e6, m, bytes / m / 1e3);
    for (int k = 0; k < K; k++) { coli_cuda_tensor_free(g[k]); coli_cuda_tensor_free(u[k]); coli_cuda_tensor_free(d[k]); free(wg[k]); free(wu[k]); free(wd[k]); }
}

int main() {
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { puts("no CUDA device: bench-gq-cuda skipped"); return 0; }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("bench-gq-cuda: %s, %d SMs, %.0f GB/s theoretical (%d-bit @ %.0f MHz), median of %d samples of 10 back-to-back launches\n",
           p.name, p.multiProcessorCount, 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9, p.memoryBusWidth, p.memoryClockRate / 1e3, iters());
    gq_ref_seed(20261017);
    puts("dense (blk_matmul through coli_cuda_tensor_gemm_async, device pointers):");
    dense("DeltaNet in_proj qkv++z", T_Q8_0, 2048, 12288);
    dense("attention q_proj", T_Q8_0, 2048, 8192);
    dense("DeltaNet out_proj / attn o_proj", T_Q8_0, 4096, 2048);
    dense("shared expert gate/up", T_Q8_0, 2048, 512);
    dense("shared expert down", T_Q8_0, 512, 2048);
    dense("shared expert gate/up (Q6_K)", T_Q6_K, 2048, 512);
    dense("expert gate (Q4_K)", T_Q4_K, 2048, 512);
    dense("expert down (Q5_K)", T_Q5_K, 512, 2048);
    dense("lm_head", T_Q6_K, 2048, 248320);
    puts("expert group (grouped_hidden_blk_dual + grouped_down_blk through issue/take):");
    group("routed experts, one token", 8, T_Q4_K, T_Q5_K, 2048, 512);
    group("routed experts, one token", 4, T_Q4_K, T_Q6_K, 2048, 512);
    group("routed experts, one token", 1, T_Q4_K, T_Q5_K, 2048, 512);
    coli_cuda_shutdown();
    return 0;
}
