/* The tensor-source façade (src.h, qwen35_names.h, gguf_xform.h) and the
 * engine's Cfg from GGUF metadata. Contract and cases:
 * 07_Tests/IntegrationTest/src_facade.md (sylph).
 *
 *   (no args)                       self-contained suite: name table both ways,
 *                                   the un-transforms on audit-shaped data,
 *                                   refusal paths -> "all passed", exit 0
 *   <model.gguf> <hf_dir> [--tol f16]
 *                                   cross-check: Cfg from the GGUF equals the Cfg
 *                                   the HF config.json implies; every tensor the
 *                                   engine loads and every expert slice equals the
 *                                   HF snapshot (bit-exact for an F32 GGUF; one
 *                                   f16 ulp with --tol f16; A_log within 4*2^-24 absolute). A refusal
 *                                   exits non-zero naming tensor and rule. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define QWEN36_NO_MAIN
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
static unsigned g_seed = 20261006;
static unsigned rnd(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }
static float frand(void) { return ((int)(rnd() & 0xFFFF) - 32768) / 32768.f; }
static float bf16r(float v) { uint32_t b; memcpy(&b, &v, 4); b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000u; memcpy(&v, &b, 4); return v; }

/* ---- forward permutation, as tools/st2gguf.py applies it ---------------------------------- */
static void perm_rows_fwd(float *dst, const float *src, int64_t nrows, int64_t cols, int64_t off, int vh, int vk, int per) {
    memcpy(dst, src, (size_t)nrows * cols * sizeof(float));
    for (int j = 0; j < vh; j++) { int h = gx_hf_head(j, vh, vk);
        memcpy(dst + (off + (int64_t)j * per) * cols, src + (off + (int64_t)h * per) * cols, (size_t)per * cols * sizeof(float)); }
}
static void perm_cols_fwd(float *dst, const float *src, int64_t nrows, int vh, int vk, int per) {
    int64_t w = (int64_t)vh * per;
    for (int64_t r = 0; r < nrows; r++) for (int j = 0; j < vh; j++)
        memcpy(dst + r * w + (int64_t)j * per, src + r * w + (int64_t)gx_hf_head(j, vh, vk) * per, (size_t)per * sizeof(float));
}

static void suite(void) {
    /* name table: every kind the engine asks for, both ways */
    static const char *kinds[] = {
        "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
        "linear_attn.conv1d.weight", "linear_attn.dt_bias", "linear_attn.A_log", "linear_attn.norm.weight", "linear_attn.out_proj.weight",
        "mlp.gate.weight", "mlp.gate.e_score_correction_bias", "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight" };
    for (size_t k = 0; k < sizeof kinds / sizeof kinds[0]; k++) {
        char hf[256], g[256], back[256]; int layer, expert;
        snprintf(hf, sizeof hf, "model.layers.%d.%s", 37, kinds[k]);
        int x = qn_to_gguf(hf, g, sizeof g, &layer, &expert);
        CHECK(x >= 0 && layer == 37 && expert == -1 && !strncmp(g, "blk.37.", 7), "%s -> gguf (%d, layer %d)", hf, x, layer);
        CHECK(qn_to_hf(g, back, sizeof back) == 0 && !strcmp(back, hf), "%s -> %s -> %s round trip", hf, g, back);
    }
    static const char *globals[] = { "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight" };
    for (int k = 0; k < 3; k++) { char g[256], back[256]; int l, e;
        CHECK(qn_to_gguf(globals[k], g, sizeof g, &l, &e) >= 0 && l == -1 && !qn_to_hf(g, back, sizeof back) && !strcmp(back, globals[k]), "global %s round trip", globals[k]); }
    { char g[256]; int l, e;
      CHECK(qn_to_gguf("model.layers.3.mlp.experts.5.up_proj.weight", g, sizeof g, &l, &e) == QN_X_EXPERT && l == 3 && e == 5 && !strcmp(g, "blk.3.ffn_up_exps.weight"), "expert slice name -> %s (layer %d expert %d)", g, l, e);
      CHECK(qn_to_gguf("model.layers.0.mlp.bogus.weight", g, sizeof g, &l, &e) == -1 && qn_to_gguf("something.else", g, sizeof g, &l, &e) == -1, "unknown names refused");
      CHECK(qn_to_hf("blk.2.nonsense", g, sizeof g) == -1, "unknown GGUF name refused"); }

    /* the permutation: audit instance and the general formula */
    { int vh = 32, vk = 16; int ok = 1;
      for (int j = 0; j < 32; j++) { int want = j < 16 ? 2 * j : 2 * (j - 16) + 1; ok &= gx_hf_head(j, vh, vk) == want; ok &= gx_gguf_head(want, vh, vk) == j; }
      CHECK(ok, "vh 32 / vk 16: GGUF head j holds HF head 2j (j<16), 2(j-16)+1 (j>=16); inverse consistent");
      for (int j = 0; j < 8; j++) ok &= gx_gguf_head(gx_hf_head(j, 8, 4), 8, 4) == j;
      CHECK(ok, "vh 8 / vk 4 (tiny): inverse consistent"); }

    /* un-transforms on audit-shaped tensors (vh 32, vk 16, vdim = kdim = 128), hidden kept small */
    { const int vh = 32, vk = 16, vdim = 128, kdim = 128, H = 256, K = 4;
      int64_t conv_dim = 2LL * vk * kdim + (int64_t)vh * vdim, vtot = (int64_t)vh * vdim;
      struct { const char *name; int64_t rows, cols, off; int per; int cols_perm; } cases[] = {
          { "attn_gate rows",     vtot,     H, 0,            vdim, 0 }, { "ssm_alpha rows", vh, H, 0, 1, 0 },
          { "attn_qkv v rows",    conv_dim, H, 2LL*vk*kdim,  vdim, 0 }, { "ssm_conv1d channels", conv_dim, K, 2LL*vk*kdim, vdim, 0 },
          { "ssm_out columns",    H,        vtot, 0,         vdim, 1 } };
      for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
          int64_t n = cases[c].rows * cases[c].cols;
          float *hf = malloc(n * 4), *gg = malloc(n * 4), *back = malloc(n * 4);
          for (int64_t i = 0; i < n; i++) hf[i] = frand();
          if (cases[c].cols_perm) { perm_cols_fwd(gg, hf, cases[c].rows, vh, vk, cases[c].per); gx_unperm_colblocks_f32(back, gg, cases[c].rows, vh, vk, cases[c].per); }
          else { perm_rows_fwd(gg, hf, cases[c].rows, cases[c].cols, cases[c].off, vh, vk, cases[c].per); gx_unperm_rowblocks_f32(back, gg, cases[c].rows, cases[c].cols, cases[c].off, vh, vk, cases[c].per); }
          CHECK(memcmp(hf, back, (size_t)n * 4) == 0 && memcmp(hf, gg, (size_t)n * 4) != 0, "%s: permuted then un-permuted bit-exact (and the permutation is not the identity)", cases[c].name);
          free(hf); free(gg); free(back);
      }
      float a[32], g[32], b[32]; for (int i = 0; i < 32; i++) a[i] = frand();
      for (int j = 0; j < 32; j++) g[j] = a[gx_hf_head(j, vh, vk)];
      gx_unperm_elems_f32(b, g, vh, vk); CHECK(memcmp(a, b, sizeof a) == 0, "per-head elements (dt_bias / ssm_a) un-permuted bit-exact");
      /* quantized rows / column blocks: Q8_0 rows of H elements (34 B per 32), column blocks of 128 = 4 blocks */
      { size_t rb = (size_t)(H / 32) * 34; int64_t rows = vtot;
        uint8_t *hf8 = malloc(rb * rows), *gg8 = malloc(rb * rows), *b8 = malloc(rb * rows);
        for (size_t i = 0; i < rb * rows; i++) hf8[i] = (uint8_t)rnd();
        for (int j = 0; j < vh; j++) memcpy(gg8 + (size_t)j * vdim * rb, hf8 + (size_t)gx_hf_head(j, vh, vk) * vdim * rb, (size_t)vdim * rb);
        gx_unperm_rowblocks(b8, gg8, rows, rb, 0, vh, vk, vdim);
        CHECK(memcmp(hf8, b8, rb * rows) == 0, "Q8_0 block rows un-permuted bit-exact");
        size_t bph = (size_t)(vdim / 32) * 34, rowb = (size_t)vh * bph;
        uint8_t *hc = malloc(rowb * H), *gc = malloc(rowb * H), *bc = malloc(rowb * H);
        for (size_t i = 0; i < rowb * H; i++) hc[i] = (uint8_t)rnd();
        for (int r = 0; r < H; r++) for (int j = 0; j < vh; j++) memcpy(gc + (size_t)r * rowb + (size_t)j * bph, hc + (size_t)r * rowb + (size_t)gx_hf_head(j, vh, vk) * bph, bph);
        CHECK(gx_unperm_colblocks_raw(bc, gc, H, vh, vk, vdim, 32, 34) == 0 && memcmp(hc, bc, rowb * H) == 0, "Q8_0 column blocks (vdim 128 = 4 blocks) un-permuted bit-exact");
        CHECK(gx_unperm_colblocks_raw(bc, gc, H, 8, 4, 8, 32, 34) == -1, "column blocks narrower than a quant block are refused (caller dequantizes)");
        free(hf8); free(gg8); free(b8); free(hc); free(gc); free(bc); }
    }
    /* norm offset and A_log */
    { int bad = 0, far = 0;
      for (int i = 0; i < 4096; i++) {
          float w = bf16r(1.f + frand() * 0.1f), st = (1.f + w); float x = st; gx_norm_unplus1(&x, 1); bad += x != w;
          float al = frand() * 2.f, a = -expf(al), y = a; gx_alog_from_a(&y, 1);
          far += fabsf(al - y) > 4.f * ldexpf(1.f, -24);
      }
      CHECK(bad == 0, "%d of 4096 bf16 norm weights do not survive 1+w -> w", bad);
      CHECK(far == 0, "%d of 4096 A_log values farther than 4*2^-24 after -exp -> log(-a)", far);
      float pos = 0.5f; CHECK(gx_alog_from_a(&pos, 1) == 1, "a non-negative ssm_a entry is reported"); }
    CHECK(!gq_supported(GGML_TYPE_Q4_1) && !gq_supported(GGML_TYPE_Q2_K) && !gq_supported(GGML_TYPE_IQ4_NL), "types outside the v1 set are unsupported");
    CHECK(gq_selftest() == 0, "gq_selftest");
}

/* ---- cross-check against the HF snapshot --------------------------------------------------------- */
static jval *cfg_json(const char *dir, char **arena) {
    char path[2304]; snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1); if (fread(buf, 1, (size_t)n, f) != (size_t)n) exit(1); buf[n] = 0; fclose(f);
    jval *r = json_parse(buf, arena);
    jval *tc = json_get(r, "text_config"); return tc && tc->t == J_OBJ ? tc : r;
}
static double tjn(jval *r, const char *k, double dflt) { jval *v = json_get(r, k); return v && v->t == J_NUM ? v->num : dflt; }

static int ulp16_ok(float a, float b) {           /* |a-b| within one f16 ulp at a's magnitude */
    if (a == b) return 1;
    float m = fabsf(a); int e; frexpf(m, &e);          /* m = f * 2^e, f in [0.5,1) */
    float ulp = ldexpf(1.f, e - 11); if (ulp < ldexpf(1.f, -24)) ulp = ldexpf(1.f, -24);
    return fabsf(a - b) <= ulp;
}
/* A_log = log(-(-exp(A_log))): the round trip's error is about ulp(a)/|a|, i.e. a few 2^-24 in absolute terms, whatever A_log's own magnitude */
static int alog_ok(float a, float b) { return fabsf(a - b) <= 4.f * ldexpf(1.f, -24); }

static int g_tol_f16 = 0;
static int64_t g_compared = 0, g_tensors = 0;
static void compare_tensor(Model *m, shards *S2, const char *hf, int alog) {
    int64_t n = st_numel(S2, hf);
    if (n < 0) { CHECK(0, "%s absent from the HF snapshot", hf); return; }
    int64_t ng = ts_numel(&m->src, hf);
    if (ng != n) { CHECK(0, "%s: %lld elements in the GGUF, %lld in the snapshot", hf, (long long)ng, (long long)n); return; }
    float *a = malloc((size_t)n * 4), *b = malloc((size_t)n * 4);
    st_read_f32(S2, hf, a, 0); ts_read_f32(&m->src, hf, b, n);
    int64_t bad = 0, first = -1;
    for (int64_t i = 0; i < n; i++) {
        int ok = alog ? alog_ok(a[i], b[i]) : g_tol_f16 ? ulp16_ok(a[i], b[i]) : memcmp(&a[i], &b[i], 4) == 0;
        if (!ok) { bad++; if (first < 0) first = i; }
    }
    if (bad) CHECK(0, "%s: %lld of %lld values differ (first at %lld: snapshot %.9g, GGUF %.9g)", hf, (long long)bad, (long long)n, (long long)first, a[first], b[first]);
    g_compared += n; g_tensors++;
    free(a); free(b);
}
static void compare_expert(Model *m, shards *S2, int layer, int eid) {
    TsExpert e; if (ts_expert(&m->src, layer, eid, &e)) { CHECK(0, "block %d: no expert slices", layer); return; }
    static const char *proj[3] = { "gate_proj", "up_proj", "down_proj" };
    for (int k = 0; k < 3; k++) {
        char hf[256]; snprintf(hf, sizeof hf, "model.layers.%d.mlp.experts.%d.%s.weight", layer, eid, proj[k]);
        int64_t n = st_numel(S2, hf); if (n != e.rows[k] * e.cols[k]) { CHECK(0, "%s: %lld elements vs slice %lldx%lld", hf, (long long)n, (long long)e.rows[k], (long long)e.cols[k]); continue; }
        uint8_t *raw = malloc(e.bytes[k]); float *a = malloc((size_t)n * 4), *b = malloc((size_t)n * 4);
        ts_read_expert_slice(&m->src, &e, k, raw); st_read_f32(S2, hf, a, 0);
        for (int64_t r = 0; r < e.rows[k]; r++) gq_deq_row(e.type[k], raw + (size_t)r * gq_row_bytes(e.type[k], (int)e.cols[k]), b + r * e.cols[k], (int)e.cols[k]);
        int64_t bad = 0; for (int64_t i = 0; i < n; i++) bad += !(g_tol_f16 ? ulp16_ok(a[i], b[i]) : memcmp(&a[i], &b[i], 4) == 0);
        if (bad) CHECK(0, "%s: %lld of %lld values differ", hf, (long long)bad, (long long)n);
        g_compared += n; g_tensors++;
        free(raw); free(a); free(b);
    }
}

static int crosscheck(const char *gguf, const char *hfdir) {
    static Model m; memset(&m, 0, sizeof m);
    if (!ts_is_gguf_path(gguf)) { fprintf(stderr, "%s is not a GGUF source\n", gguf); return 1; }
    ts_init(&m.src, &m.S, gguf, NULL);
    cfg_from_gguf(&m);                                   /* exits with the rule on a refusal */
    validate_cfg(&m.c, m.c.n_layers);
    Cfg *c = &m.c;
    char *arena = NULL; jval *r = cfg_json(hfdir, &arena);
    int H = (int)tjn(r, "hidden_size", 0), L = (int)tjn(r, "num_hidden_layers", 0), V = (int)tjn(r, "vocab_size", 0);
    int qh = (int)tjn(r, "num_attention_heads", 0), kvh = (int)tjn(r, "num_key_value_heads", 0), hd = (int)tjn(r, "head_dim", H / (qh ? qh : 1));
    int E = (int)tjn(r, "num_experts", 0), K = (int)tjn(r, "num_experts_per_tok", 0), F = (int)tjn(r, "moe_intermediate_size", 0), Fs = (int)tjn(r, "shared_expert_intermediate_size", F);
    int vh = (int)tjn(r, "linear_num_value_heads", 0), vk = (int)tjn(r, "linear_num_key_heads", 0), kd = (int)tjn(r, "linear_key_head_dim", 0), vd = (int)tjn(r, "linear_value_head_dim", 0), ck = (int)tjn(r, "linear_conv_kernel_dim", 0);
    double eps = tjn(r, "rms_norm_eps", 1e-6), theta = tjn(r, "rope_theta", 10000.0), prf = tjn(r, "partial_rotary_factor", 0.25);
    jval *rp = json_get(r, "rope_parameters"); if (rp && rp->t == J_OBJ) { theta = tjn(rp, "rope_theta", theta); prf = tjn(rp, "partial_rotary_factor", prf); }
    jval *gate = json_get(r, "attn_output_gate"); int og = gate && gate->t == J_BOOL ? gate->boolean : 1;
    jval *ntp = json_get(r, "norm_topk_prob"); int norm_topk = ntp && ntp->t == J_BOOL ? ntp->boolean : 0;
    jval *lt = json_get(r, "layer_types");
    CHECK(c->hidden == H && c->n_layers == L && c->vocab == V, "Cfg hidden/n_layers/vocab %d/%d/%d vs config %d/%d/%d", c->hidden, c->n_layers, c->vocab, H, L, V);
    CHECK(c->q_heads == qh && c->kv_heads == kvh && c->head_dim == hd && c->k_head_dim == hd && c->v_head_dim == hd, "Cfg heads %d/%d dim %d vs config %d/%d/%d", c->q_heads, c->kv_heads, c->head_dim, qh, kvh, hd);
    CHECK(c->q_head_dim == (og ? 2 * hd : hd) && c->attn_output_gate == og && c->o_in == qh * hd, "Cfg q_head_dim %d gate %d o_in %d vs config (%d, %d, %d)", c->q_head_dim, c->attn_output_gate, c->o_in, og ? 2 * hd : hd, og, qh * hd);
    CHECK(c->rotary_dim == (int)(hd * prf + 0.5) && fabs(c->partial_rotary_factor - prf) < 1e-6 && c->rope_dim == hd, "Cfg rotary_dim %d factor %g vs config %d / %g", c->rotary_dim, c->partial_rotary_factor, (int)(hd * prf + 0.5), prf);
    CHECK(fabs(c->theta - theta) < 1e-3 * theta && fabs(c->eps - eps) < 1e-12, "Cfg theta %g eps %g vs config %g %g", c->theta, c->eps, theta, eps);
    CHECK(c->n_experts == E && c->topk == K && c->inter == F && c->shared_inter == Fs, "Cfg experts %d/%d inter %d/%d vs config %d/%d/%d/%d", c->n_experts, c->topk, c->inter, c->shared_inter, E, K, F, Fs);
    CHECK(c->norm_topk == norm_topk && c->n_group == 1 && c->topk_group == 1, "Cfg norm_topk %d n_group %d topk_group %d", c->norm_topk, c->n_group, c->topk_group);
    CHECK(c->dn_vheads == vh && c->dn_kheads == vk && c->dn_kdim == kd && c->dn_vdim == vd && c->dn_convk == ck && c->dn_conv_dim == 2 * vk * kd + vh * vd,
          "Cfg DeltaNet %d/%d/%d/%d/%d/%d vs config %d/%d/%d/%d/%d/%d", c->dn_vheads, c->dn_kheads, c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim, vh, vk, kd, vd, ck, 2 * vk * kd + vh * vd);
    int kinds_ok = lt && lt->t == J_ARR && lt->len == L;
    for (int i = 0; kinds_ok && i < L; i++) kinds_ok = c->is_attn[i] == (lt->kids[i]->t == J_STR && !strcmp(lt->kids[i]->str, "full_attention"));
    CHECK(kinds_ok, "Cfg is_attn[] equals config layer_types");
    CHECK(c->has_qk_norm == 1, "Cfg has_qk_norm (the snapshot carries q_norm/k_norm)");
    printf("Cfg from GGUF: %d checks run\n", 11);

    shards S2; memset(&S2, 0, sizeof S2); st_init_multi(&S2, hfdir, NULL);
    compare_tensor(&m, &S2, "model.embed_tokens.weight", 0);
    compare_tensor(&m, &S2, "model.norm.weight", 0);
    compare_tensor(&m, &S2, "lm_head.weight", 0);
    for (int i = 0; i < c->n_layers; i++) {
        char nm[256];
#define T(suffix, alog) do { snprintf(nm, sizeof nm, "model.layers.%d." suffix, i); compare_tensor(&m, &S2, nm, alog); } while (0)
        T("input_layernorm.weight", 0); T("post_attention_layernorm.weight", 0); T("mlp.gate.weight", 0);
        T("mlp.shared_expert.gate_proj.weight", 0); T("mlp.shared_expert.up_proj.weight", 0); T("mlp.shared_expert.down_proj.weight", 0);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.shared_expert_gate.weight", i); if (st_has(&S2, nm)) compare_tensor(&m, &S2, nm, 0);
        if (c->is_attn[i]) {
            T("self_attn.q_proj.weight", 0); T("self_attn.k_proj.weight", 0); T("self_attn.v_proj.weight", 0); T("self_attn.o_proj.weight", 0);
            T("self_attn.q_norm.weight", 0); T("self_attn.k_norm.weight", 0);
        } else {
            T("linear_attn.in_proj_qkv.weight", 0); T("linear_attn.in_proj_z.weight", 0); T("linear_attn.in_proj_a.weight", 0); T("linear_attn.in_proj_b.weight", 0);
            T("linear_attn.conv1d.weight", 0); T("linear_attn.dt_bias", 0); T("linear_attn.A_log", 1); T("linear_attn.norm.weight", 0); T("linear_attn.out_proj.weight", 0);
        }
#undef T
        for (int e = 0; e < c->n_experts; e++) compare_expert(&m, &S2, i, e);
    }
    printf("compared %lld tensors / slices, %lld values, tolerance %s\n", (long long)g_tensors, (long long)g_compared, g_tol_f16 ? "one f16 ulp" : "bit-exact");
    free(arena);
    return 0;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--tol") && i + 1 < argc && !strcmp(argv[i + 1], "f16")) g_tol_f16 = 1;
    if (argc >= 3 && argv[1][0] != '-') crosscheck(argv[1], argv[2]);
    else suite();
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    printf("all passed\n"); return 0;
}
