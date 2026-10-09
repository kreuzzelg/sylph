/* The qwen36 engine with the fake CUDA backend linked in (sylph phase 5).
 * Contract: 07_Tests/IntegrationTest/cuda_tier_kquant.md, cases 3-5 and 7.
 *
 * Same command line and environment as ./qwen36; the difference is that
 * COLI_CUDA=1 starts the VRAM tier on tests/qwen36_fake_cuda.h, which in its
 * compute mode runs the block formats through gq_dot_row_ref and the dense
 * int8 path through a plain loop. The runner (run_cuda_tier_kquant.py) drives
 * this binary and the real qwen36 on the same GGUF and compares token ids and
 * PPL dumps; what this file adds is the one line the runner parses at exit:
 *
 *   [fake-cuda] uploads N · block fmts n24/n28/n29/n30 · issues N · expert rows N · matmuls N
 *
 * Include order as test_qwen36_tier_int8_engine.c: the engine first (it never
 * references coli_cuda_*), then the fake backend, then the tier, so the tier's
 * statics live in this TU and the fake's counters are visible to the report. */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

#include "../compat.h"
#include "qwen36_fake_cuda.h"
#include "../qwen36_tier.c"

static void fake_report(void) {
    fprintf(stderr, "[fake-cuda] uploads %d · block fmts %d/%d/%d/%d · issues %d · expert rows %d · matmuls %d\n",
            fake_uploads, fake_block_uploads[0], fake_block_uploads[1], fake_block_uploads[2], fake_block_uploads[3],
            fake_issues, fake_issue_rows, fake_matmuls);
}

int main(int argc, char **argv) {
    fake_block_compute = 1;    /* block formats: gq_dot_row_ref per row, host expf in the SiLU */
    fake_dense_compute = 1;    /* fmt 1 trunk matrices: computed, as test_qwen36_trunk_dense does */
    atexit(fake_report);       /* registered before the engine's atexit(qt_shutdown): runs after it */
    return qwen36_main_unused(argc, argv);
}
