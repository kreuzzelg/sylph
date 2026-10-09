/* The VRAM expert tier fed with the GGUF slot flavour, on the fake backend
 * (sylph phase 5). Contract: 07_Tests/IntegrationTest/cuda_tier_kquant.md,
 * case 2. No GPU, no toolkit.
 *
 * What a tier test can prove and what it cannot: it pins the API the engine
 * calls (qt_init_gguf refuses by type, qt_note_kq stages three raw slices and
 * uploads them under fmt 16 + type with no scale array, the budget is charged
 * per slot footprint), and -- with the fake computing the block formats through
 * gq_dot_row_ref -- that issue/take on resident experts reproduces gq_moe_run
 * BIT FOR BIT, which is the engine's summation-order contract (routed experts
 * in rank order, fmaf, shared expert afterwards). Whether the real kernels
 * reproduce gq_dot_row_ref is tests/test_gq_cuda.cu on silicon (part B). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../compat.h"   /* setenv: MinGW has none */
#include "../gq.h"
#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int fails;
static void check(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) fails++; }

enum { NL = 2, NE = 4, D = 256, IH = 256, TOPK = 2 };

static unsigned g_seed = 777;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static uint16_t rand_f16(void) { float v = (0.0005f + (rnd() & 1023) / 20000.f) * ((rnd() & 1) ? 1.f : -1.f); return gq_f32_to_f16(v); }
static void fill_blocks(int type, int nb, uint8_t *dst) {
    int ts = gq_types[type].tsize;
    for (int b = 0; b < nb; b++) {
        uint8_t *p = dst + (size_t)b * ts;
        for (int i = 0; i < ts; i++) p[i] = (uint8_t)rnd();
        switch (type) {
        case GQ_Q8_0: gq_st16(p, rand_f16()); break;
        case GQ_Q4_K: case GQ_Q5_K: gq_st16(p, rand_f16()); gq_st16(p + 2, rand_f16()); break;
        case GQ_Q6_K: gq_st16(p + 208, rand_f16()); break;
        default: break;
        }
    }
}

/* one GGUF-flavour slot: gate|up as Q4_K [IH][D], down as Q6_K [D][IH], one slab */
typedef struct { uint8_t *kq; int ktype[3]; size_t kbytes[3]; } KqSlot;
static void make_slot(KqSlot *s) {
    s->ktype[0] = GQ_Q4_K; s->ktype[1] = GQ_Q4_K; s->ktype[2] = GQ_Q6_K;
    s->kbytes[0] = gq_row_bytes(GQ_Q4_K, D) * IH; s->kbytes[1] = s->kbytes[0]; s->kbytes[2] = gq_row_bytes(GQ_Q6_K, IH) * D;
    s->kq = malloc(s->kbytes[0] + s->kbytes[1] + s->kbytes[2]);
    fill_blocks(GQ_Q4_K, (int)(s->kbytes[0] / 144), s->kq);
    fill_blocks(GQ_Q4_K, (int)(s->kbytes[1] / 144), s->kq + s->kbytes[0]);
    fill_blocks(GQ_Q6_K, (int)(s->kbytes[2] / 210), s->kq + s->kbytes[0] + s->kbytes[1]);
}

static int wait_resident(int layer, int eid) {
    for (int i = 0; i < 1000; i++) {
        if (qt_is_resident(layer, eid)) return 1;
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
    }
    return 0;
}
/* the tier keeps host storage across init/shutdown until teardown cleanup lands (as test_qwen36_tier_scale_budget) */
static void teardown(void) {
    qt_shutdown();
    free(G.slot); G.slot = NULL; free(G.is_x); G.is_x = NULL;
    free(G.fill_order); G.fill_order = NULL; free(G.heat0); G.heat0 = NULL;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1); setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1); setenv("HEAT_FILE", "", 1);
    setenv("COLI_PLACE", "off", 1); setenv("QT_UPLOAD_SYNC", "1", 1);
    setenv("CUDA_EXPERT_GB", "1", 1);
    fake_ndev = 1;

    KqSlot slots[NL][NE];
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) make_slot(&slots[l][e]);
    const size_t slot_bytes[3] = { slots[0][0].kbytes[0], slots[0][0].kbytes[1], slots[0][0].kbytes[2] };
    const uint32_t types_ok = (1u << GQ_Q4_K) | (1u << GQ_Q6_K);

    printf("(a) init by type\n");
    check(qt_init_gguf(NL, NE, D, IH, NE, TOPK, slot_bytes, types_ok) == 1, "Q4_K + Q6_K experts: tier starts");
    check(qt_ready(), "qt_ready after qt_init_gguf");
    teardown();
    check(qt_init_gguf(NL, NE, D, IH, NE, TOPK, slot_bytes, types_ok | (1u << GQ_Q4_0)) == 0, "a Q4_0 expert type is refused by name (note on stderr)");
    check(!qt_ready(), "refused tier is not ready");
    check(qt_init_gguf(NL, NE, D, IH, NE - 1, TOPK, slot_bytes, types_ok) == 0, "cap != n_experts still refused");

    printf("(b) qt_note_kq uploads three raw slices\n");
    fake_uploads = 0; memset(fake_block_uploads, 0, sizeof fake_block_uploads);
    check(qt_init_gguf(NL, NE, D, IH, NE, TOPK, slot_bytes, types_ok) == 1, "tier starts again");
    qt_note_kq(0, 0, slots[0][0].kq, slots[0][0].ktype, slots[0][0].kbytes);
    qt_fill_wait();
    check(wait_resident(0, 0), "expert (0,0) becomes resident");
    check(fake_uploads == 3, "three uploads for one expert");
    check(fake_block_uploads[1] == 2 && fake_block_uploads[3] == 1 && fake_block_uploads[0] == 0 && fake_block_uploads[2] == 0,
          "fmts 28, 28, 30 (Q4_K gate/up, Q6_K down)");
    check(last_fmt == 30 && last_bytes == slot_bytes[2], "the last upload is the down slice, its bytes exactly");
    check(last_sc == NULL, "no scale array travels with a block tensor");
    check(qs(0, 0)->tg && qs(0, 0)->tu && qs(0, 0)->td, "slot holds three device tensors");

    printf("(c) budget: footprint per slot, planned count\n");
    teardown();
    size_t exp = dev_alloc_footprint(slot_bytes[0]) + dev_alloc_footprint(slot_bytes[1]) + dev_alloc_footprint(slot_bytes[2]);
    char gb[64]; snprintf(gb, sizeof gb, "%.17g", (5 * exp + exp / 2) / 1073741824.0);
    setenv("CUDA_EXPERT_GB", gb, 1);
    check(qt_init_gguf(NL, NE, D, IH, NE, TOPK, slot_bytes, types_ok) == 1, "tier starts with a 5.5-expert allowance");
    check(G.exp_bytes == exp, "an expert is charged the sum of its three slice footprints");
    int pl[NL * NE], pe[NL * NE]; int n = qt_plan_fill(pl, pe, NL * NE);
    check(n == 5, "qt_plan_fill plans exactly 5 experts");
    check(G.used[0] == 5 * exp && G.used[0] <= G.budget[0], "used == 5 footprints, within the budget");
    teardown();
    setenv("CUDA_EXPERT_GB", "1", 1);

    printf("(d) issue/take == gq_moe_run bit for bit\n");
    fake_block_compute = 1; fake_issues = 0;
    check(qt_init_gguf(NL, NE, D, IH, NE, TOPK, slot_bytes, types_ok) == 1, "tier starts (compute mode)");
    for (int e = 0; e < NE; e++) qt_note_kq(1, e, slots[1][e].kq, slots[1][e].ktype, slots[1][e].kbytes);
    qt_fill_wait();
    int all = 1; for (int e = 0; e < NE; e++) all &= wait_resident(1, e);
    check(all, "every expert of layer 1 resident");
    float x[D]; for (int i = 0; i < D; i++) x[i] = ((int)(rnd() & 0xFFFF) - 32768) / 8192.f;
    int idx[TOPK] = { 2, 0 }; float val[TOPK] = { 0.6f, 0.4f };
    uint32_t mask = qt_issue(1, idx, TOPK, x);
    check(mask == 3u, "both routed experts handled by the GPU");
    check(fake_issues == 1 && fake_issue_rows == TOPK, "one group issue with two rows");
    float out_gpu[D]; memset(out_gpu, 0, sizeof out_gpu);
    check(qt_take(mask, val, TOPK, out_gpu) == 1, "qt_take collects");
    /* the CPU path on the same bytes */
    GqExpert ex[TOPK]; const GqExpert *exp_[TOPK];
    for (int k = 0; k < TOPK; k++) {
        KqSlot *s = &slots[1][idx[k]];
        ex[k].g = s->kq; ex[k].u = s->kq + s->kbytes[0]; ex[k].d = s->kq + s->kbytes[0] + s->kbytes[1];
        ex[k].tg = s->ktype[0]; ex[k].tu = s->ktype[1]; ex[k].td = s->ktype[2]; exp_[k] = &ex[k];
    }
    void *scratch = malloc(gq_moe_scratch_bytes(1, TOPK, D, IH));
    float out_cpu[D]; gq_moe_run(out_cpu, x, 1, TOPK, D, IH, idx, val, exp_, scratch);
    int same = memcmp(out_cpu, out_gpu, sizeof out_cpu) == 0;
    if (!same) { int worst = 0; float dm = 0; for (int d = 0; d < D; d++) { float dd = fabsf(out_cpu[d] - out_gpu[d]); if (dd > dm) { dm = dd; worst = d; } }
                 printf("    first/worst difference at d=%d: cpu %.9g gpu %.9g\n", worst, out_cpu[worst], out_gpu[worst]); }
    check(same, "tier output == gq_moe_run output, bit for bit (rank order, fmaf)");
    /* a miss stays a miss: an expert never noted is not in the mask */
    int idx2[TOPK] = { 3, 1 }; qt_note_kq(0, 3, slots[0][3].kq, slots[0][3].ktype, slots[0][3].kbytes); /* layer 0: only (0,3) offered now */
    qt_fill_wait(); wait_resident(0, 3);
    uint32_t m2 = qt_issue(0, idx2, TOPK, x);
    check(m2 == 1u, "layer 0: expert 3 resident, expert 1 a CPU miss");
    float o2[D]; memset(o2, 0, sizeof o2); check(qt_take(m2, val, TOPK, o2) == 1, "partial take collects");
    free(scratch);
    teardown();
    fake_block_compute = 0;

    printf("(e) NFR-5: the int8 container path is untouched\n");
    fake_uploads = 0;
    check(qt_init(1, NE, 64, 32, NE, TOPK, 0, 0) == 1, "int8 per-row tier starts");
    { static int8_t g[32 * 64], u[32 * 64], dn[64 * 32]; static float gs[32], us[32], ds[64];
      for (int i = 0; i < 32 * 64; i++) { g[i] = (int8_t)(i % 7 - 3); u[i] = (int8_t)(i % 5 - 2); dn[i] = (int8_t)(i % 3 - 1); }
      for (int i = 0; i < 32; i++) gs[i] = us[i] = 1.f; for (int i = 0; i < 64; i++) ds[i] = 1.f;
      qt_note(0, 1, (const uint8_t *)g, (const uint8_t *)u, (const uint8_t *)dn, gs, us, ds); qt_fill_wait(); wait_resident(0, 1); }
    check(fake_uploads == 3 && last_fmt == 1, "int8 expert still uploads as fmt 1");
    teardown();

    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) free(slots[l][e].kq);
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    puts("tier kq: all passed");
    return 0;
}
