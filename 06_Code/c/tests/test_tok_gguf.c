/* The engine tokenizer built from a GGUF's tokenizer.ggml.* arrays must equal
 * the one built from tokenizer.json (07_Tests/IntegrationTest/src_facade.md,
 * case 5).
 *
 *   (no args)                                   self-contained: the tiny tokenizer
 *                                               of test_qwen36_tokenizer.c as JSON
 *                                               and as a GGUF written here
 *   <model.gguf> <tokenizer.json> [corpus.txt] [--llama-tokenize <bin>]
 *                                               real metadata vs the real JSON on
 *                                               every line of the corpus */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* drop the loaded tokenizer so the second constructor starts clean (the engine's
 * own destroy helper lives behind COLI_EDGE_ADAPTER) */
static void tok_reset(void) {
    for (int i = 0; i < g_tok_n; i++) free(g_tok[i]);
    free(g_tok); g_tok = NULL; g_tok_n = 0;
    for (int slot = 0; slot < g_merge.cap; slot++) if (g_merge.used && g_merge.used[slot]) free(g_merge.keys[slot]);
    free(g_rev.keys); free(g_rev.vals); free(g_rev.used);
    free(g_merge.keys); free(g_merge.vals); free(g_merge.used);
    memset(&g_rev, 0, sizeof g_rev); memset(&g_merge, 0, sizeof g_merge);
    for (int i = 0; i < g_nspecial; i++) free(g_sp_str[i]);
    free(g_sp_str); free(g_sp_id); free(g_sp_len);
    g_sp_str = NULL; g_sp_id = NULL; g_sp_len = NULL; g_nspecial = 0;
}

typedef struct { char **line; int n; } Lines;
static Lines read_lines(const char *path) {
    Lines L = { NULL, 0 };
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1); if (fread(buf, 1, (size_t)n, f) != (size_t)n) exit(1); buf[n] = 0; fclose(f);
    int cap = 64; L.line = malloc(sizeof(char *) * cap);
    char *p = buf;
    while (*p) {
        char *e = strchr(p, '\n'); size_t len = e ? (size_t)(e - p) : strlen(p);
        if (L.n == cap) { cap *= 2; L.line = realloc(L.line, sizeof(char *) * cap); }
        L.line[L.n] = malloc(len + 1); memcpy(L.line[L.n], p, len); L.line[L.n][len] = 0; L.n++;
        if (!e) break; p = e + 1;
    }
    free(buf); return L;
}
static const char *builtin[] = { "X", "X<|im_end|>", "X.", "X.<|im_end|>", "X}<|im_end|>", "X!<|im_end|>", "X.\n<|im_end|>", "X  Y", "X<|im_end|><|im_end|>", "<|im_end|>X", "  ", "end_Y" };

/* a minimal GGUF v3 with only the tokenizer arrays */
static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void put_str(FILE *f, const char *s) { put_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }
static void put_key_str(FILE *f, const char *k, const char *v) { put_str(f, k); put_u32(f, 8); put_str(f, v); }
static void write_tok_gguf(const char *path, const char **tokens, const int *types, int n, const char **merges, int nm) {
    FILE *f = fopen(path, "wb"); if (!f) exit(1);
    fwrite("GGUF", 1, 4, f); put_u32(f, 3); put_u64(f, 0); put_u64(f, 5);
    put_key_str(f, "tokenizer.ggml.model", "gpt2"); put_key_str(f, "tokenizer.ggml.pre", "qwen35");
    put_str(f, "tokenizer.ggml.tokens"); put_u32(f, 9); put_u32(f, 8); put_u64(f, (uint64_t)n); for (int i = 0; i < n; i++) put_str(f, tokens[i]);
    put_str(f, "tokenizer.ggml.token_type"); put_u32(f, 9); put_u32(f, 5); put_u64(f, (uint64_t)n); for (int i = 0; i < n; i++) put_u32(f, (uint32_t)types[i]);
    put_str(f, "tokenizer.ggml.merges"); put_u32(f, 9); put_u32(f, 8); put_u64(f, (uint64_t)nm); for (int i = 0; i < nm; i++) put_str(f, merges[i]);
    long pos = ftell(f); while (pos % 32) { fputc(0, f); pos++; }
    fclose(f);
}

typedef struct { int *ids; int n; char *text; } Enc;
static Enc encode_line(const char *s) {
    Enc e = { NULL, 0, NULL }; encode_text(s, &e.ids, &e.n);
    e.text = malloc(1 << 16); if (e.n > 0) decode_range(e.ids, 0, e.n, e.text, 1 << 16); else e.text[0] = 0;
    return e;
}

int main(int argc, char **argv) {
    const char *gguf = NULL, *json = NULL, *corpus = NULL, *llama = NULL, *encode = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--llama-tokenize") && i + 1 < argc) llama = argv[++i];
        else if (!strcmp(argv[i], "--encode") && i + 1 < argc) encode = argv[++i];
        else if (!gguf) gguf = argv[i]; else if (!json) json = argv[i]; else corpus = argv[i];
    }
    if (encode) {
        /* sylph FR-32 tokenizer gate: `test_tok_gguf <gguf> --encode <text>` prints the ids
         * of the whole file (one per line) with the GGUF's tokenizer, so the equivalence
         * harness can compare them with `llama-tokenize --ids` on the same bytes. */
        if (!gguf) { fprintf(stderr, "usage: test_tok_gguf <gguf> --encode <text file>\n"); return 2; }
        FILE *f = fopen(encode, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", encode); return 1; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *text = malloc((size_t)n + 1); if (!text || fread(text, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", encode); return 1; }
        text[n] = 0; fclose(f);
        GgufSet G; gguf_open_set_or_die(&G, gguf, NULL);
        load_tokenizer_gguf(&G);
        if (!g_tok) { fprintf(stderr, "FAIL: GGUF tokenizer did not load\n"); return 1; }
        int *ids = NULL, nid = 0; encode_text(text, &ids, &nid);
        for (int i = 0; i < nid; i++) printf("%d\n", ids[i]);
        return 0;
    }
    char tmp_json[64] = "", tmp_gguf[64] = "";
    Lines L = { NULL, 0 };
    if (!gguf) {
        /* the tiny tokenizer of test_qwen36_tokenizer.c, as JSON and as GGUF arrays */
        const char *js =
            "{\"model\":{\"vocab\":{\"X\":0,\".\":1,\"<\":2,\"|\":3,\"i\":4,\"m\":5,\"Y\":6,"
            "\"}\":8,\"\\\"\":9,\"\xC4\x8A\":10,\"\xC4\xA0\":11,\"\xC4\xA0\xC4\xA0\":12,"
            "\"_\":13,\"e\":14,\"n\":15,\"d\":16,\">\":17,\"!\":18,\"\xC4\xA0Y\":19},"
            "\"merges\":[\"\xC4\xA0 \xC4\xA0\",\"\xC4\xA0 Y\"]},"
            "\"added_tokens\":[{\"id\":7,\"content\":\"<|im_end|>\",\"special\":true}]}";
        snprintf(tmp_json, sizeof tmp_json, "test_tok_gguf_%d.json", (int)getpid()); snprintf(tmp_gguf, sizeof tmp_gguf, "test_tok_gguf_%d.gguf", (int)getpid());
        FILE *f = fopen(tmp_json, "wb"); fwrite(js, 1, strlen(js), f); fclose(f);
        const char *tokens[20] = { "X", ".", "<", "|", "i", "m", "Y", "<|im_end|>", "}", "\"", "\xC4\x8A", "\xC4\xA0", "\xC4\xA0\xC4\xA0", "_", "e", "n", "d", ">", "!", "\xC4\xA0Y" };
        int types[20]; for (int i = 0; i < 20; i++) types[i] = i == 7 ? 3 : 1;
        const char *merges[2] = { "\xC4\xA0 \xC4\xA0", "\xC4\xA0 Y" };
        write_tok_gguf(tmp_gguf, tokens, types, 20, merges, 2);
        gguf = tmp_gguf; json = tmp_json;
        L.n = (int)(sizeof builtin / sizeof builtin[0]); L.line = malloc(sizeof(char *) * L.n); for (int i = 0; i < L.n; i++) L.line[i] = strdup(builtin[i]);
    } else if (corpus) L = read_lines(corpus);
    else { L.n = (int)(sizeof builtin / sizeof builtin[0]); L.line = malloc(sizeof(char *) * L.n); for (int i = 0; i < L.n; i++) L.line[i] = strdup(builtin[i]); }

    load_tokenizer(json);
    if (!g_tok) { printf("FAIL: tokenizer.json did not load\n"); return 1; }
    int n_json = g_tok_n;
    Enc *A = malloc(sizeof(Enc) * L.n);
    for (int i = 0; i < L.n; i++) A[i] = encode_line(L.line[i]);
    tok_reset();

    GgufSet G; gguf_open_set_or_die(&G, gguf, NULL);
    load_tokenizer_gguf(&G);
    if (!g_tok) { printf("FAIL: GGUF tokenizer did not load\n"); return 1; }
    /* llama.cpp pads the token list to the embedding rows with [PADn] placeholders
     * (type unused): the GGUF vocab may be longer, and every extra id must be one */
    int extra_real = 0; for (int i = n_json; i < g_tok_n; i++) if (g_tok[i] && strncmp(g_tok[i], "[PAD", 4)) extra_real++;
    CHECK(g_tok_n >= n_json && extra_real == 0, "vocab: GGUF %d ids, JSON %d; %d of the extra ids are not [PADn] placeholders", g_tok_n, n_json, extra_real);
    int bad_ids = 0, bad_text = 0;
    for (int i = 0; i < L.n; i++) {
        Enc B = encode_line(L.line[i]);
        int same = B.n == A[i].n && (A[i].n == 0 || !memcmp(A[i].ids, B.ids, sizeof(int) * A[i].n));
        if (!same) { bad_ids++; if (bad_ids <= 5) { printf("FAIL: line %d %.60s%s\n  json:", i, L.line[i], strlen(L.line[i]) > 60 ? "…" : ""); for (int k = 0; k < A[i].n; k++) printf(" %d", A[i].ids[k]); printf("\n  gguf:"); for (int k = 0; k < B.n; k++) printf(" %d", B.ids[k]); printf("\n"); } }
        if (strcmp(A[i].text, B.text)) { bad_text++; if (bad_text <= 3) printf("FAIL: line %d decodes differ: %.80s | %.80s\n", i, A[i].text, B.text); }
        free(B.ids); free(B.text);
    }
    CHECK(bad_ids == 0, "%d of %d lines encode to different ids", bad_ids, L.n);
    CHECK(bad_text == 0, "%d of %d lines decode differently", bad_text, L.n);
    if (llama) {
        /* owner's machine: ids from llama-tokenize --ids on each line (one process per line, stdlib popen) */
        int bad_l = 0;
        for (int i = 0; i < L.n; i++) {
            char cmd[8192]; FILE *pf;
            char tf[64]; snprintf(tf, sizeof tf, "test_tok_gguf_line_%d.txt", (int)getpid());
            FILE *lf = fopen(tf, "wb"); fwrite(L.line[i], 1, strlen(L.line[i]), lf); fclose(lf);
            snprintf(cmd, sizeof cmd, "\"%s\" -m \"%s\" -f \"%s\" --ids --no-bos 2>/dev/null", llama, gguf, tf);
            pf = popen(cmd, "r"); if (!pf) { printf("FAIL: cannot run %s\n", llama); bad_l++; break; }
            char out[1 << 16] = ""; size_t got = fread(out, 1, sizeof out - 1, pf); out[got] = 0; pclose(pf); remove(tf);
            /* "[1, 2, 3]" */
            int ids[8192], n = 0; for (char *p = out; *p && n < 8192; ) { while (*p && (*p < '0' || *p > '9')) p++; if (!*p) break; ids[n++] = (int)strtol(p, &p, 10); }
            int same = n == A[i].n && !memcmp(ids, A[i].ids, sizeof(int) * (size_t)n);
            if (!same) { bad_l++; if (bad_l <= 5) printf("FAIL: line %d differs from llama-tokenize (%d vs %d ids)\n", i, n, A[i].n); }
        }
        CHECK(bad_l == 0, "%d of %d lines differ from llama-tokenize", bad_l, L.n);
    }
    printf("%d lines, vocab %d\n", L.n, g_tok_n);
    gguf_close(&G);
    if (tmp_json[0]) { remove(tmp_json); remove(tmp_gguf); }
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    printf("all passed\n"); return 0;
}
