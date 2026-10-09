/* The CUDA block-format predicates (sylph phase 5), unit-tested WITHOUT a GPU
 * or a CUDA toolchain. Contract: 07_Tests/IntegrationTest/cuda_tier_kquant.md,
 * "Block formats in the CUDA backend".
 *
 * A raw ggml block tensor reaches the device under fmt = 16 + ggml type id:
 * Q8_0 -> 24, Q4_K -> 28, Q5_K -> 29, Q6_K -> 30. These tensors have their own
 * kernel branch (as fmt 6 and 7 do) and never route through weight_at, so the
 * generic predicate coli_cuda_weight_at_supported keeps refusing them and the
 * absorb gates stay closed. The host-side gates that admit a block tensor to an
 * upload or a launch consult coli_cuda_block_fmt_supported, and the two helpers
 * give the type id and the block length the upload size check uses.
 *
 * ENUMERATED, NOT SAMPLED: every value a descriptor could present, -8..64. */
#include "../backend_cuda.h"

#include <stdio.h>

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

int main(void) {
    CHECK(COLI_FMT_GGML_BASE == 16);
    for (int fmt = -8; fmt <= 64; fmt++) {
        int block = fmt == 24 || fmt == 28 || fmt == 29 || fmt == 30;
        if (coli_cuda_block_fmt_supported(fmt) != block) { printf("FAIL block_fmt_supported(%d) = %d\n", fmt, coli_cuda_block_fmt_supported(fmt)); fails++; }
        int type = fmt == 24 ? 8 : fmt == 28 ? 12 : fmt == 29 ? 13 : fmt == 30 ? 14 : -1;
        if (coli_cuda_block_fmt_type(fmt) != type) { printf("FAIL block_fmt_type(%d) = %d\n", fmt, coli_cuda_block_fmt_type(fmt)); fails++; }
        int elems = fmt == 24 ? 32 : block ? 256 : 0;
        if (coli_cuda_block_fmt_elems(fmt) != elems) { printf("FAIL block_fmt_elems(%d) = %d\n", fmt, coli_cuda_block_fmt_elems(fmt)); fails++; }
        /* the generic decoder's set is untouched: block formats are not in it */
        if (block && coli_cuda_weight_at_supported(fmt)) { printf("FAIL weight_at_supported(%d) admits a block format\n", fmt); fails++; }
    }
    /* the pre-existing truth table (test_cuda_fmt_guard.c) still holds */
    CHECK(coli_cuda_weight_at_supported(0) && coli_cuda_weight_at_supported(1) && coli_cuda_weight_at_supported(2));
    CHECK(coli_cuda_weight_at_supported(3) && coli_cuda_weight_at_supported(4) && coli_cuda_weight_at_supported(8));
    CHECK(!coli_cuda_weight_at_supported(5) && !coli_cuda_weight_at_supported(6) && !coli_cuda_weight_at_supported(7));
    /* ggml ids that stay OFF the device in v1 (Q4_0 = 2 -> 18; Q2_K..Q3_K, Q8_K, IQ*) */
    CHECK(!coli_cuda_block_fmt_supported(16 + 2) && !coli_cuda_block_fmt_supported(16 + 10) && !coli_cuda_block_fmt_supported(16 + 15));

    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    puts("cuda block fmt guard: all passed");
    return 0;
}
