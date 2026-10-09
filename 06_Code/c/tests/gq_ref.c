/* CPU reference for tests/test_gq_cuda.cu (sylph phase 5, cuda_tier_kquant.md part B).
 *
 * gq.h is C (static inline kernels with the fp-contract pragmas and the AVX2 /
 * NEON intrinsics the engine compiles); compiling the reference HERE as C and
 * exposing a few symbols keeps the CUDA test comparing against the SAME code
 * the engine runs -- the tests/mxfp4_ref.c arrangement. The random block
 * generator is the one tests/test_gq_kernels.c uses (finite f16 scale fields,
 * random payload), so both oracles see the same distribution of blocks. */
#include "../gq.h"

static unsigned g_seed = 12345;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static float frand(void) { return ((int)(rnd() & 0xFFFF) - 32768) / 8192.f; }   /* [-4, 4) */

void gq_ref_seed(unsigned s) { g_seed = s; }
int gq_ref_supported(int type) { return gq_supported(type); }
int gq_ref_block_elems(int type) { return gq_supported(type) ? gq_types[type].block : 0; }
int gq_ref_block_bytes(int type) { return gq_supported(type) ? gq_types[type].tsize : 0; }
size_t gq_ref_row_bytes(int type, int I) { return gq_row_bytes(type, I); }
const char *gq_ref_type_name(int type) { return gq_type_name(type); }

/* y = row . x with the scalar reference (the bit pattern every kernel must reproduce) */
float gq_ref_dot(int type, const uint8_t *row, const float *x, int I) { return gq_dot_row_ref(type, row, x, I); }
/* the engine's fastest CPU path (bit-identical to the reference per gq_kernels.md) */
float gq_ref_dot_fast(int type, const uint8_t *row, const float *x, int I) { return gq_dot_row(type, row, x, I); }
/* exact dequantization of one row (f32), for the tolerance checks */
int gq_ref_deq_row(int type, const uint8_t *row, float *out, int I) { return gq_deq_row(type, row, out, I); }

static uint16_t rand_f16(int wide) {
    if (wide) return (uint16_t)(((rnd() & 1) << 15) | ((rnd() % 31) << 10) | (rnd() & 0x3FF));
    float v = (0.0005f + (rnd() & 1023) / 20000.f) * ((rnd() & 1) ? 1.f : -1.f);   /* |v| in [5e-4, 0.052) */
    return gq_f32_to_f16(v);
}
/* nb blocks of `type` with random payload and finite scale fields */
void gq_ref_fill_blocks(int type, int nb, uint8_t *dst, int wide) {
    int ts = gq_types[type].tsize;
    for (int b = 0; b < nb; b++) {
        uint8_t *p = dst + (size_t)b * ts;
        for (int i = 0; i < ts; i++) p[i] = (uint8_t)rnd();
        switch (type) {
        case GQ_Q8_0: gq_st16(p, rand_f16(wide)); break;
        case GQ_Q4_K: case GQ_Q5_K: gq_st16(p, rand_f16(wide)); gq_st16(p + 2, rand_f16(wide)); break;
        case GQ_Q6_K: gq_st16(p + 208, rand_f16(wide)); break;
        default: break;
        }
    }
}
void gq_ref_fill_x(float *x, int n) { for (int i = 0; i < n; i++) x[i] = frand(); }
