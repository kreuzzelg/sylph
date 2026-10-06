/* qwen35_names.h — HF ⇄ GGUF tensor names for llama.cpp's `qwen35moe`
 * (Qwen3.5 / Qwen3.6 MoE), plus the load-time un-transform each tensor needs.
 *
 * The engine (qwen36.c) asks for tensors by the HF names its converter keeps,
 * `model.layers.<i>.<kind>` and three globals. A GGUF written by llama.cpp's
 * converter holds the same values under `blk.<i>.<name>`, some of them
 * transformed (architecture §7, audited 2026-10-05): RMSNorm weights as 1+w,
 * `ssm_a` as −exp(A_log), and the DeltaNet value heads permuted. This table is
 * the single place that knows both spellings and the transform kind; src.h
 * applies the inverse (gguf_xform.h), tools/st2gguf.py applies the forward.
 * Header-only, static, no I/O. */
#ifndef COLI_QWEN35_NAMES_H
#define COLI_QWEN35_NAMES_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

enum {
    QN_X_NONE = 0,
    QN_X_NORM_PLUS1,      /* stored 1+w: subtract 1 */
    QN_X_SSM_A,           /* stored −exp(A_log) (per value head, permuted): A_log = log(−a), un-permute */
    QN_X_PERM_HEAD_ROWS,  /* rows are value heads × vdim (attn_gate) or × 1 (ssm_alpha/beta): un-permute row blocks */
    QN_X_PERM_HEAD_ELEMS, /* one element per value head (ssm_dt.bias): un-permute elements */
    QN_X_PERM_QKV_ROWS,   /* value third of attn_qkv rows (after 2·vk·kdim q/k rows): un-permute row blocks */
    QN_X_PERM_CONV_ROWS,  /* ssm_conv1d {k, conv_dim}: channels of the value third: un-permute row blocks */
    QN_X_PERM_HEAD_COLS,  /* ssm_out [hidden][vh·vdim]: input columns by value head: un-permute column blocks */
    QN_X_EXPERT,          /* slice e of a 3-D expert tensor */
};

typedef struct { const char *hf; const char *gguf; int xform; } QnEntry;

/* layer kinds: HF suffix after "model.layers.<i>." ⇄ GGUF suffix after "blk.<i>." */
static const QnEntry qn_layer[] = {
    { "input_layernorm.weight",                 "attn_norm.weight",            QN_X_NORM_PLUS1 },
    { "post_attention_layernorm.weight",        "post_attention_norm.weight",  QN_X_NORM_PLUS1 },
    { "self_attn.q_proj.weight",                "attn_q.weight",               QN_X_NONE },
    { "self_attn.k_proj.weight",                "attn_k.weight",               QN_X_NONE },
    { "self_attn.v_proj.weight",                "attn_v.weight",               QN_X_NONE },
    { "self_attn.o_proj.weight",                "attn_output.weight",          QN_X_NONE },
    { "self_attn.q_norm.weight",                "attn_q_norm.weight",          QN_X_NORM_PLUS1 },
    { "self_attn.k_norm.weight",                "attn_k_norm.weight",          QN_X_NORM_PLUS1 },
    { "linear_attn.in_proj_qkv.weight",         "attn_qkv.weight",             QN_X_PERM_QKV_ROWS },
    { "linear_attn.in_proj_z.weight",           "attn_gate.weight",            QN_X_PERM_HEAD_ROWS },
    { "linear_attn.in_proj_a.weight",           "ssm_alpha.weight",            QN_X_PERM_HEAD_ROWS },
    { "linear_attn.in_proj_b.weight",           "ssm_beta.weight",             QN_X_PERM_HEAD_ROWS },
    { "linear_attn.conv1d.weight",              "ssm_conv1d.weight",           QN_X_PERM_CONV_ROWS },
    { "linear_attn.dt_bias",                    "ssm_dt.bias",                 QN_X_PERM_HEAD_ELEMS },
    { "linear_attn.A_log",                      "ssm_a",                       QN_X_SSM_A },
    { "linear_attn.norm.weight",                "ssm_norm.weight",             QN_X_NONE },
    { "linear_attn.out_proj.weight",            "ssm_out.weight",              QN_X_PERM_HEAD_COLS },
    { "mlp.gate.weight",                        "ffn_gate_inp.weight",         QN_X_NONE },
    { "mlp.gate.e_score_correction_bias",       "exp_probs_b.bias",            QN_X_NONE },
    { "mlp.shared_expert.gate_proj.weight",     "ffn_gate_shexp.weight",       QN_X_NONE },
    { "mlp.shared_expert.up_proj.weight",       "ffn_up_shexp.weight",         QN_X_NONE },
    { "mlp.shared_expert.down_proj.weight",     "ffn_down_shexp.weight",       QN_X_NONE },
    { "mlp.shared_expert_gate.weight",          "ffn_gate_inp_shexp.weight",   QN_X_NONE },
};
/* per-expert HF kinds ⇄ the 3-D GGUF tensors they are slices of */
static const QnEntry qn_expert[] = {
    { "gate_proj.weight", "ffn_gate_exps.weight", QN_X_EXPERT },
    { "up_proj.weight",   "ffn_up_exps.weight",   QN_X_EXPERT },
    { "down_proj.weight", "ffn_down_exps.weight", QN_X_EXPERT },
};
static const QnEntry qn_global[] = {
    { "model.embed_tokens.weight", "token_embd.weight",  QN_X_NONE },
    { "model.norm.weight",         "output_norm.weight", QN_X_NORM_PLUS1 },
    { "lm_head.weight",            "output.weight",      QN_X_NONE },
};
#define QN_N_LAYER  ((int)(sizeof qn_layer  / sizeof qn_layer[0]))
#define QN_N_EXPERT ((int)(sizeof qn_expert / sizeof qn_expert[0]))
#define QN_N_GLOBAL ((int)(sizeof qn_global / sizeof qn_global[0]))

/* "model.layers.<i>.<rest>" → i and rest; -1 if not a layer name */
static inline int qn_split_layer(const char *hf, const char **rest) {
    const char *p = "model.layers.";
    size_t n = strlen(p);
    if (strncmp(hf, p, n)) return -1;
    char *end; long i = strtol(hf + n, &end, 10);
    if (end == hf + n || *end != '.' || i < 0 || i > 100000) return -1;
    *rest = end + 1;
    return (int)i;
}

/* HF (engine) name → GGUF name. Returns the transform kind (QN_X_*), with
 * *layer (-1 for globals) and *expert (-1 unless a per-expert slice); -1 if
 * the name is not in the table. */
static inline int qn_to_gguf(const char *hf, char *out, size_t cap, int *layer, int *expert) {
    *layer = -1; *expert = -1;
    for (int k = 0; k < QN_N_GLOBAL; k++)
        if (!strcmp(hf, qn_global[k].hf)) { snprintf(out, cap, "%s", qn_global[k].gguf); return qn_global[k].xform; }
    const char *rest; int i = qn_split_layer(hf, &rest);
    if (i < 0) return -1;
    *layer = i;
    for (int k = 0; k < QN_N_LAYER; k++)
        if (!strcmp(rest, qn_layer[k].hf)) { snprintf(out, cap, "blk.%d.%s", i, qn_layer[k].gguf); return qn_layer[k].xform; }
    const char *ep = "mlp.experts.";
    if (!strncmp(rest, ep, strlen(ep))) {
        char *end; long e = strtol(rest + strlen(ep), &end, 10);
        if (end != rest + strlen(ep) && *end == '.' && e >= 0 && e <= 100000) {
            for (int k = 0; k < QN_N_EXPERT; k++)
                if (!strcmp(end + 1, qn_expert[k].hf)) { *expert = (int)e; snprintf(out, cap, "blk.%d.%s", i, qn_expert[k].gguf); return QN_X_EXPERT; }
        }
    }
    return -1;
}

/* GGUF name → HF (engine) name; for messages and tests. Expert tensors map to
 * "model.layers.<i>.mlp.experts.*.<proj>.weight". Returns 0, -1 if unknown. */
static inline int qn_to_hf(const char *gguf, char *out, size_t cap) {
    for (int k = 0; k < QN_N_GLOBAL; k++)
        if (!strcmp(gguf, qn_global[k].gguf)) { snprintf(out, cap, "%s", qn_global[k].hf); return 0; }
    if (strncmp(gguf, "blk.", 4)) return -1;
    char *end; long i = strtol(gguf + 4, &end, 10);
    if (end == gguf + 4 || *end != '.') return -1;
    const char *rest = end + 1;
    for (int k = 0; k < QN_N_LAYER; k++)
        if (!strcmp(rest, qn_layer[k].gguf)) { snprintf(out, cap, "model.layers.%ld.%s", i, qn_layer[k].hf); return 0; }
    for (int k = 0; k < QN_N_EXPERT; k++)
        if (!strcmp(rest, qn_expert[k].gguf)) { snprintf(out, cap, "model.layers.%ld.mlp.experts.*.%s", i, qn_expert[k].hf); return 0; }
    return -1;
}

#endif /* COLI_QWEN35_NAMES_H */
