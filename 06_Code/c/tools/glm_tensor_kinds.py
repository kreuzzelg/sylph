"""Every tensor name a GLM-5.2 (glm_moe_dsa) checkpoint can carry, classified.

The contract of tools/st2gguf.py --arch glm-dsa (07_Tests/IntegrationTest/glm_assembly.md):
``classify`` names every tensor or raises ``UnknownTensor``; the converter stops on the
first name it cannot place, so a container is never quietly missing or quietly carrying a
tensor. Torch-free. The C twin of the name table is c/glm_names.h.

Layout (transformers GlmMoeDsaForCausalLM, prefix ``model.``):
  * globals: embed_tokens, norm, lm_head;
  * every block: MLA attention (q_a/q_a_layernorm/q_b, kv_a_proj_with_mqa/kv_a_layernorm/
    kv_b_proj, o_proj), two norms, the DSA indexer (wq_b, wk, weights_proj, k_norm.{weight,bias});
  * leading dense blocks: mlp.{gate,up,down}_proj; MoE blocks: mlp.gate.{weight,
    e_score_correction_bias}, mlp.shared_experts.*, mlp.experts.<e>.*;
  * the NextN block (index num_hidden_layers, when num_nextn_predict_layers = 1): eh_proj,
    enorm, hnorm, shared_head.norm (+ shared_head.head, tied to lm_head and skipped) and a
    full MoE block.
"""
import re

ATTN_KINDS = frozenset((
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.o_proj.weight",
))
INDEXER_KINDS = frozenset((
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
))
DENSE_MLP_KINDS = frozenset(("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"))
MOE_KINDS = frozenset((
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
))
NEXTN_KINDS = frozenset(("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"))
LAYER_KINDS = ATTN_KINDS | INDEXER_KINDS | DENSE_MLP_KINDS | MOE_KINDS | NEXTN_KINDS
GLOBAL_KINDS = frozenset(("embed_tokens.weight", "norm.weight", "lm_head.weight"))

_EXPERT = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_SCALE = re.compile(r".*\.(weight_scale_inv|scale_inv|weight_scale)$")

# deliberately not converted (counted, reported): the tied MTP head and rotary buffers
SKIP_KINDS = frozenset(("shared_head.head.weight",))
SKIP_SUFFIXES = (".rotary_emb.inv_freq",)

PREFIX = "model."


class UnknownTensor(KeyError):
    """A name the contract does not place. The converter must stop on it."""


def classify(name):
    """("global", kind) | ("layer", index, kind) | ("skip", why); raise UnknownTensor.
    Expert tensors keep their normalised kind "mlp.experts.<e>.<proj>.weight"."""
    if name == "lm_head.weight" or name == PREFIX + "lm_head.weight":
        return ("global", "lm_head.weight")
    if name.endswith(SKIP_SUFFIXES):
        return ("skip", "rotary buffer")
    if _SCALE.match(name):
        return ("skip", "fp8 scale (dequantize the checkpoint first; the converter reads f32/bf16)")
    if name.startswith(PREFIX):
        rest = name[len(PREFIX):]
        if rest in GLOBAL_KINDS:
            return ("global", rest)
        m = _LAYER.match(rest)
        if m:
            kind = m.group(2)
            if kind in SKIP_KINDS:
                return ("skip", "shared_head.head (tied to lm_head)")
            if kind in LAYER_KINDS or _EXPERT.match(kind):
                return ("layer", int(m.group(1)), kind)
    raise UnknownTensor(name)


def expert_index(kind):
    m = _EXPERT.match(kind)
    return (int(m.group(1)), m.group(2)) if m else None
