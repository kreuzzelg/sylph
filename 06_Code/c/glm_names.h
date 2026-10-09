/* glm_names.h — HF ⇄ GGUF tensor names for llama.cpp's `glm-dsa` (GLM-5 / 5.2),
 * the C twin of tools/glm_tensor_kinds.py (sylph phase 6, architecture v1 §6.1,
 * 07_Tests/IntegrationTest/glm_assembly.md).
 *
 * The engine (colibri.c) asks for tensors by the HF names its converter keeps,
 * `model.layers.<i>.<kind>` and three globals; a GGUF written by llama.cpp's
 * converter holds the same values under `blk.<i>.<name>`. GLM stores nothing
 * transformed (no 1+w norms, no head permutation), with one structural change:
 * `kv_b_proj` is written as the absorbed split `attn_k_b` (per head transposed)
 * + `attn_v_b`, or by some converters fused as `attn_kv_b`. The two split
 * spellings map BACK to kv_b_proj here; the engine reconciles them
 * (glm_kv_b_from_split in colibri.c). The NextN block lives at i = n_layers:
 * its head tensors are `blk.<L>.nextn.*`, its block tensors ordinary `blk.<L>.*`.
 * Header-only, static, no I/O. Reuses the QN_X_* transform ids of qwen35_names.h
 * (only QN_X_NONE and QN_X_EXPERT occur). */
#ifndef COLI_GLM_NAMES_H
#define COLI_GLM_NAMES_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "qwen35_names.h"

/* layer kinds: HF suffix after "model.layers.<i>." ⇄ GGUF suffix after "blk.<i>." */
static const QnEntry glm_layer[] = {
    { "input_layernorm.weight",                    "attn_norm.weight",            QN_X_NONE },
    { "post_attention_layernorm.weight",           "ffn_norm.weight",             QN_X_NONE },
    { "self_attn.q_a_proj.weight",                 "attn_q_a.weight",             QN_X_NONE },
    { "self_attn.q_a_layernorm.weight",            "attn_q_a_norm.weight",        QN_X_NONE },
    { "self_attn.q_b_proj.weight",                 "attn_q_b.weight",             QN_X_NONE },
    { "self_attn.kv_a_proj_with_mqa.weight",       "attn_kv_a_mqa.weight",        QN_X_NONE },
    { "self_attn.kv_a_layernorm.weight",           "attn_kv_a_norm.weight",       QN_X_NONE },
    { "self_attn.kv_b_proj.weight",                "attn_kv_b.weight",            QN_X_NONE },   /* policy 1 (fused) spelling */
    { "self_attn.o_proj.weight",                   "attn_output.weight",          QN_X_NONE },
    { "mlp.gate_proj.weight",                      "ffn_gate.weight",             QN_X_NONE },   /* leading dense blocks */
    { "mlp.up_proj.weight",                        "ffn_up.weight",               QN_X_NONE },
    { "mlp.down_proj.weight",                      "ffn_down.weight",             QN_X_NONE },
    { "mlp.gate.weight",                           "ffn_gate_inp.weight",         QN_X_NONE },
    { "mlp.gate.e_score_correction_bias",          "exp_probs_b.bias",            QN_X_NONE },
    { "mlp.shared_experts.gate_proj.weight",       "ffn_gate_shexp.weight",       QN_X_NONE },
    { "mlp.shared_experts.up_proj.weight",         "ffn_up_shexp.weight",         QN_X_NONE },
    { "mlp.shared_experts.down_proj.weight",       "ffn_down_shexp.weight",       QN_X_NONE },
    { "self_attn.indexer.wq_b.weight",             "indexer.attn_q_b.weight",     QN_X_NONE },
    { "self_attn.indexer.wk.weight",               "indexer.attn_k.weight",       QN_X_NONE },
    { "self_attn.indexer.weights_proj.weight",     "indexer.proj.weight",         QN_X_NONE },
    { "self_attn.indexer.k_norm.weight",           "indexer.k_norm.weight",       QN_X_NONE },
    { "self_attn.indexer.k_norm.bias",             "indexer.k_norm.bias",         QN_X_NONE },
    /* NextN head (block i = n_layers) */
    { "eh_proj.weight",                            "nextn.eh_proj.weight",        QN_X_NONE },
    { "enorm.weight",                              "nextn.enorm.weight",          QN_X_NONE },
    { "hnorm.weight",                              "nextn.hnorm.weight",          QN_X_NONE },
    { "shared_head.norm.weight",                   "nextn.shared_head_norm.weight", QN_X_NONE },
};
/* the absorbed split (policy 2): two GGUF names, one HF name; never produced by HF→GGUF here */
static const QnEntry glm_split[] = {
    { "self_attn.kv_b_proj.weight", "attn_k_b.weight", QN_X_NONE },
    { "self_attn.kv_b_proj.weight", "attn_v_b.weight", QN_X_NONE },
};
static const QnEntry glm_global[] = {
    { "model.embed_tokens.weight", "token_embd.weight",  QN_X_NONE },
    { "model.norm.weight",         "output_norm.weight", QN_X_NONE },
    { "lm_head.weight",            "output.weight",      QN_X_NONE },
};
#define GLM_N_LAYER  ((int)(sizeof glm_layer  / sizeof glm_layer[0]))
#define GLM_N_SPLIT  ((int)(sizeof glm_split  / sizeof glm_split[0]))
#define GLM_N_GLOBAL ((int)(sizeof glm_global / sizeof glm_global[0]))

/* HF (engine) name → GGUF name. Returns the transform kind (QN_X_NONE, or
 * QN_X_EXPERT for a per-expert slice with *expert set), *layer (-1 for globals);
 * -1 if the name is not in the table. Same contract as qn_to_gguf (src.h). */
static inline int glm_to_gguf(const char *hf, char *out, size_t cap, int *layer, int *expert) {
    *layer = -1; *expert = -1;
    for (int k = 0; k < GLM_N_GLOBAL; k++)
        if (!strcmp(hf, glm_global[k].hf)) { snprintf(out, cap, "%s", glm_global[k].gguf); return glm_global[k].xform; }
    const char *rest; int i = qn_split_layer(hf, &rest);
    if (i < 0) return -1;
    *layer = i;
    for (int k = 0; k < GLM_N_LAYER; k++)
        if (!strcmp(rest, glm_layer[k].hf)) { snprintf(out, cap, "blk.%d.%s", i, glm_layer[k].gguf); return glm_layer[k].xform; }
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

/* GGUF name → HF (engine) name; attn_k_b / attn_v_b map back to kv_b_proj. 0 / -1. */
static inline int glm_to_hf(const char *gguf, char *out, size_t cap) {
    for (int k = 0; k < GLM_N_GLOBAL; k++)
        if (!strcmp(gguf, glm_global[k].gguf)) { snprintf(out, cap, "%s", glm_global[k].hf); return 0; }
    if (strncmp(gguf, "blk.", 4)) return -1;
    char *end; long i = strtol(gguf + 4, &end, 10);
    if (end == gguf + 4 || *end != '.') return -1;
    const char *rest = end + 1;
    for (int k = 0; k < GLM_N_LAYER; k++)
        if (!strcmp(rest, glm_layer[k].gguf)) { snprintf(out, cap, "model.layers.%ld.%s", i, glm_layer[k].hf); return 0; }
    for (int k = 0; k < GLM_N_SPLIT; k++)
        if (!strcmp(rest, glm_split[k].gguf)) { snprintf(out, cap, "model.layers.%ld.%s", i, glm_split[k].hf); return 0; }
    for (int k = 0; k < QN_N_EXPERT; k++)
        if (!strcmp(rest, qn_expert[k].gguf)) { snprintf(out, cap, "model.layers.%ld.mlp.experts.*.%s", i, qn_expert[k].hf); return 0; }
    return -1;
}

/* The two names the test contract uses (glm_assembly.md case 1): 0 on success. */
static inline int glm_gguf_name(const char *hf, char *out, size_t cap) { int l, e; return glm_to_gguf(hf, out, cap, &l, &e) < 0 ? -1 : 0; }
static inline int glm_hf_name(const char *gguf, char *out, size_t cap) { return glm_to_hf(gguf, out, cap); }

#endif /* COLI_GLM_NAMES_H */
