/* The GLM-5.2 (glm-dsa) arm of the tensor-source façade in colibri.c (sylph
 * phase 6). Contract and cases: 07_Tests/IntegrationTest/glm_assembly.md.
 *
 *   (no args)                       self-contained suite: the name table of
 *                                   glm_names.h both ways, the MLA reconciliation
 *                                   on synthetic data (kv_b split as the converter
 *                                   stores it, rebuilt by the engine's routine),
 *                                   idx_type[] derivation, the NextN precision
 *                                   predicate, refusals -> "all passed", exit 0
 *   <model.gguf> <hf_dir> [--tol q8] cross-check: Cfg from the GGUF equals the Cfg
 *                                   config.json implies; every dense tensor the
 *                                   engine loads, kv_b rebuilt from attn_k_b/attn_v_b
 *                                   and every expert slice equal the HF snapshot
 *                                   (bit-exact for an F32 GGUF; with --tol q8 the
 *                                   Q8_0 tensors are compared as the exact dequant of
 *                                   the same bytes). A refusal exits non-zero naming
 *                                   tensor and rule.
 *
 * Mirrors tests/test_gguf_load.c (the qwen36 arm). The engine is included whole so
 * the test reads the Model the loader built, not a re-statement of it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define COLIBRI_NO_MAIN
#define main colibri_main_unused
#include "../colibri.c"
#undef main

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
static unsigned g_seed = 20261009;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static float frand(void) { return ((int)(rnd() & 0xFFFF) - 32768) / 32768.f; }

/* ---- the converter's split of kv_b (what llama.cpp stores), on synthetic data ---------- */
/* kv_b [H*(nope+v)][kvl]  ->  k_b {nope, kvl, H}: per head K_h[kvl][nope] = (k rows)^T
 *                              v_b {kvl, v, H}:    per head V_h[v][kvl]   = v rows as stored */
static void split_kv_b(const float *kv_b, int H, int nope, int v, int kvl, float *k_b, float *v_b) {
    for (int h = 0; h < H; h++) {
        const float *K = kv_b + (size_t)h * (nope + v) * kvl, *V = K + (size_t)nope * kvl;
        for (int r = 0; r < kvl; r++) for (int j = 0; j < nope; j++) k_b[((size_t)h * kvl + r) * nope + j] = K[(size_t)j * kvl + r];
        memcpy(v_b + (size_t)h * v * kvl, V, (size_t)v * kvl * sizeof(float));
    }
}

static void suite(void) {
    /* 1. name table both ways (glm_names.h) */
    static const char *pairs[][2] = {
        {"model.embed_tokens.weight", "token_embd.weight"}, {"model.norm.weight", "output_norm.weight"}, {"lm_head.weight", "output.weight"},
        {"model.layers.3.input_layernorm.weight", "blk.3.attn_norm.weight"}, {"model.layers.3.post_attention_layernorm.weight", "blk.3.ffn_norm.weight"},
        {"model.layers.3.self_attn.q_a_proj.weight", "blk.3.attn_q_a.weight"}, {"model.layers.3.self_attn.q_a_layernorm.weight", "blk.3.attn_q_a_norm.weight"},
        {"model.layers.3.self_attn.q_b_proj.weight", "blk.3.attn_q_b.weight"}, {"model.layers.3.self_attn.kv_a_proj_with_mqa.weight", "blk.3.attn_kv_a_mqa.weight"},
        {"model.layers.3.self_attn.kv_a_layernorm.weight", "blk.3.attn_kv_a_norm.weight"}, {"model.layers.3.self_attn.o_proj.weight", "blk.3.attn_output.weight"},
        {"model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"}, {"model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"}, {"model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"},
        {"model.layers.3.mlp.gate.weight", "blk.3.ffn_gate_inp.weight"}, {"model.layers.3.mlp.gate.e_score_correction_bias", "blk.3.exp_probs_b.bias"},
        {"model.layers.3.mlp.shared_experts.gate_proj.weight", "blk.3.ffn_gate_shexp.weight"}, {"model.layers.3.mlp.shared_experts.up_proj.weight", "blk.3.ffn_up_shexp.weight"},
        {"model.layers.3.mlp.shared_experts.down_proj.weight", "blk.3.ffn_down_shexp.weight"},
        {"model.layers.3.self_attn.indexer.wq_b.weight", "blk.3.indexer.attn_q_b.weight"}, {"model.layers.3.self_attn.indexer.wk.weight", "blk.3.indexer.attn_k.weight"},
        {"model.layers.3.self_attn.indexer.weights_proj.weight", "blk.3.indexer.proj.weight"}, {"model.layers.3.self_attn.indexer.k_norm.weight", "blk.3.indexer.k_norm.weight"},
        {"model.layers.3.self_attn.indexer.k_norm.bias", "blk.3.indexer.k_norm.bias"},
        {"model.layers.5.eh_proj.weight", "blk.5.nextn.eh_proj.weight"}, {"model.layers.5.enorm.weight", "blk.5.nextn.enorm.weight"},
        {"model.layers.5.hnorm.weight", "blk.5.nextn.hnorm.weight"}, {"model.layers.5.shared_head.norm.weight", "blk.5.nextn.shared_head_norm.weight"},
    };
    char buf[256];
    for (size_t i = 0; i < sizeof pairs / sizeof *pairs; i++) {
        CHECK(glm_gguf_name(pairs[i][0], buf, sizeof buf) == 0 && !strcmp(buf, pairs[i][1]), "HF->GGUF %s -> %s (got %s)", pairs[i][0], pairs[i][1], buf);
        CHECK(glm_hf_name(pairs[i][1], buf, sizeof buf) == 0 && !strcmp(buf, pairs[i][0]), "GGUF->HF %s -> %s (got %s)", pairs[i][1], pairs[i][0], buf);
    }
    /* kv_b_proj: the one HF name with two GGUF spellings (policy 1 fused, policy 2 split) */
    CHECK(glm_gguf_name("model.layers.3.self_attn.kv_b_proj.weight", buf, sizeof buf) == 0 && !strcmp(buf, "blk.3.attn_kv_b.weight"), "kv_b_proj maps to attn_kv_b (fused spelling)");
    CHECK(glm_hf_name("blk.3.attn_k_b.weight", buf, sizeof buf) == 0 && !strcmp(buf, "model.layers.3.self_attn.kv_b_proj.weight"), "attn_k_b maps back to kv_b_proj");
    CHECK(glm_hf_name("blk.3.attn_v_b.weight", buf, sizeof buf) == 0 && !strcmp(buf, "model.layers.3.self_attn.kv_b_proj.weight"), "attn_v_b maps back to kv_b_proj");
    CHECK(glm_gguf_name("model.layers.3.self_attn.rotary_emb.inv_freq", buf, sizeof buf) != 0, "a name outside the table is refused");
    CHECK(glm_hf_name("blk.3.ffn_norm_exps.weight", buf, sizeof buf) != 0, "an unknown GGUF name is refused");

    /* 2. MLA reconciliation: the engine rebuilds kv_b from the split bit for bit (f32) */
    { enum { H = 3, NOPE = 24, V = 32, KVL = 32 };
      size_t n = (size_t)H * (NOPE + V) * KVL;
      float *kv = malloc(n * sizeof(float)), *kb = malloc((size_t)H * KVL * NOPE * sizeof(float)), *vb = malloc((size_t)H * V * KVL * sizeof(float)), *out = malloc(n * sizeof(float));
      for (size_t i = 0; i < n; i++) kv[i] = frand();
      split_kv_b(kv, H, NOPE, V, KVL, kb, vb);
      CHECK(glm_kv_b_from_split(out, kb, vb, H, NOPE, V, KVL) == 0, "glm_kv_b_from_split accepts the geometry");
      CHECK(memcmp(out, kv, n * sizeof(float)) == 0, "kv_b rebuilt from attn_k_b/attn_v_b is bit-exact");
      CHECK(glm_kv_b_from_split(out, kb, vb, H, NOPE + 1, V, KVL) != 0, "a geometry mismatch is refused");
      free(kv); free(kb); free(vb); free(out); }

    /* 3. idx_type[] derivation: explicit array wins, else tensor presence; both agree on a consistent file */
    { int8_t from_presence[6] = {1, 1, 1, 1, 1, 0}, explicit_types[6] = {1, 1, 1, 1, 1, 0}, derived[6];
      uint8_t has_indexer[6] = {1, 1, 1, 1, 1, 0};
      glm_idx_types(derived, 6, NULL, has_indexer);
      CHECK(memcmp(derived, from_presence, 6) == 0, "idx_type from tensor presence");
      glm_idx_types(derived, 6, explicit_types, has_indexer);
      CHECK(memcmp(derived, explicit_types, 6) == 0, "idx_type from the explicit indexer.types array");
      has_indexer[2] = 0;
      CHECK(glm_idx_types(derived, 6, explicit_types, has_indexer) != 0, "types array naming a block without indexer tensors is refused"); }

    /* 4. the NextN precision predicate (v1 §7.5): >= 8 bpw loads, below needs MTP=1 */
    CHECK(glm_mtp_bits_ok(GQ_Q8_0, 0) && glm_mtp_bits_ok(GQ_F16, 0) && glm_mtp_bits_ok(GQ_BF16, 0) && glm_mtp_bits_ok(GQ_F32, 0), "Q8_0/F16/BF16/F32 eh_proj load");
    CHECK(!glm_mtp_bits_ok(GQ_Q6_K, 0) && !glm_mtp_bits_ok(GQ_Q5_K, 0) && !glm_mtp_bits_ok(GQ_Q4_K, 0), "K-quant eh_proj is skipped by default");
    CHECK(glm_mtp_bits_ok(GQ_Q4_K, 1), "MTP=1 overrides the guard");
}

static int cross_check(const char *gguf, const char *hf, int tol_q8) {
    (void)tol_q8;
    Model m;
    if (glm_cross_check(&m, gguf, hf, tol_q8, &fails) != 0) { printf("FAIL: cross-check aborted\n"); return 1; }
    return fails ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc >= 3) { int tol = argc >= 5 && !strcmp(argv[3], "--tol") && !strcmp(argv[4], "q8"); if (cross_check(argv[1], argv[2], tol)) { printf("%d failure(s)\n", fails); return 1; } puts("cross-check passed"); return 0; }
    suite();
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    puts("all passed");
    return 0;
}
