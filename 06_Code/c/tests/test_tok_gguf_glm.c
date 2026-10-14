/* The GLM engine's tokenizer built from a GGUF's tokenizer.ggml.* arrays (colibri.c,
 * glm_tok_load, sylph phase 6) must equal the one tok.h builds from tokenizer.json
 * (07_Tests/IntegrationTest/gguf_hardening.md, case 5; the GLM half of FR-20).
 *
 *   <model.gguf> <tokenizer.json> <corpus.txt>   both constructors on every line of the
 *                                                corpus: identical ids, identical decode;
 *                                                the control tokens resolve by name in both
 *   <model.gguf> --encode <text file>            print the ids (one line), the FR-32 gate
 *   <model.gguf> --expect-refusal                 the GGUF's tokenizer.ggml.pre is not GLM's:
 *                                                the constructor must refuse (exit 1) and
 *                                                name the key -- this mode exits 0 iff it did
 *
 * Mirrors tests/test_tok_gguf.c (the qwen36 arm). The engine is included whole. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#define COLIBRI_NO_MAIN
#include "../colibri.c"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

typedef struct { char **line; int n; } Lines;
static Lines read_lines(const char *path) {
    Lines L = { NULL, 0 };
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1); if (fread(buf, 1, (size_t)n, f) != (size_t)n) exit(1); buf[n] = 0; fclose(f);
    for (char *p = buf; *p;) {
        char *e = strchr(p, '\n'); if (e) *e = 0;
        L.line = realloc(L.line, sizeof(char *) * (size_t)(L.n + 1)); L.line[L.n++] = p;
        if (!e) break; p = e + 1;
    }
    return L;
}

static int same(Tok *A, Tok *B, const char *text, int *na, int *nb, int *ia, int *ib, int cap) {
    *na = tok_encode(A, text, (int)strlen(text), ia, cap);
    *nb = tok_encode(B, text, (int)strlen(text), ib, cap);
    return *na == *nb && !memcmp(ia, ib, (size_t)*na * sizeof(int));
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_tok_gguf_glm <model.gguf> <tokenizer.json> <corpus.txt> | <model.gguf> --encode <text> | <model.gguf> --expect-refusal\n"); return 2; }
    GgufSet G;
    if (gguf_open_set(&G, argv[1], NULL)) { fprintf(stderr, "gguf: %s\n", G.err); return 1; }
    if (!strcmp(argv[2], "--expect-refusal")) {
        /* the constructor exits on a foreign pre-tokenizer: run it in a child and read the verdict */
        fflush(stdout); fflush(stderr);
        pid_t pid = fork();
        if (pid == 0) { Tok T; glm_tok_load(&T, &G); _exit(0); }
        int st = 0; waitpid(pid, &st, 0);
        int refused = WIFEXITED(st) && WEXITSTATUS(st) == 1;
        printf("%s\n", refused ? "refused as expected" : "FAIL: a foreign tokenizer.ggml.pre was accepted");
        return refused ? 0 : 1;
    }
    Tok Gt; glm_tok_load(&Gt, &G);
    if (!strcmp(argv[2], "--encode")) {
        if (argc < 4) { fprintf(stderr, "usage: test_tok_gguf_glm <gguf> --encode <text file>\n"); return 2; }
        Lines L = read_lines(argv[3]); int ids[65536];
        for (int i = 0; i < L.n; i++) { int n = tok_encode(&Gt, L.line[i], (int)strlen(L.line[i]), ids, 65536); for (int k = 0; k < n; k++) printf("%d%s", ids[k], k + 1 < n ? " " : ""); printf("\n"); }
        return 0;
    }
    Tok Jt; tok_load(&Jt, argv[2]);
    CHECK(Gt.n_ids == Jt.n_ids, "id space: GGUF %d, JSON %d", Gt.n_ids, Jt.n_ids);
    CHECK(Gt.nsp == Jt.nsp, "added tokens: GGUF %d, JSON %d", Gt.nsp, Jt.nsp);
    CHECK(Gt.o200k == Jt.o200k && Gt.gpt2 == Jt.gpt2 && Gt.kimi == Jt.kimi, "pre-tokenizer family: GGUF o200k=%d gpt2=%d kimi=%d, JSON %d/%d/%d", Gt.o200k, Gt.gpt2, Gt.kimi, Jt.o200k, Jt.gpt2, Jt.kimi);
    static const char *ctrl[] = { "<|endoftext|>", "<|user|>", "<|assistant|>", "[gMASK]", "<sop>" };
    for (size_t k = 0; k < sizeof ctrl / sizeof *ctrl; k++)
        CHECK(tok_id_of(&Gt, ctrl[k]) == tok_id_of(&Jt, ctrl[k]), "control token %s: GGUF id %d, JSON id %d", ctrl[k], tok_id_of(&Gt, ctrl[k]), tok_id_of(&Jt, ctrl[k]));
    Lines L = read_lines(argv[3]); int ia[65536], ib[65536], na, nb, lines = 0, toks = 0;
    for (int i = 0; i < L.n; i++) {
        if (!same(&Gt, &Jt, L.line[i], &na, &nb, ia, ib, 65536)) {
            fails++; printf("FAIL: line %d differs (%d vs %d ids): %.60s\n", i + 1, na, nb, L.line[i]);
            for (int k = 0; k < (na < nb ? na : nb); k++) if (ia[k] != ib[k]) { printf("      first difference at %d: GGUF %d JSON %d\n", k, ia[k], ib[k]); break; }
        }
        for (int k = 0; k < na; k++) {
            const char *a = ia[k] >= 0 && ia[k] < Gt.n_ids ? Gt.id2str[ia[k]] : NULL, *b = ia[k] >= 0 && ia[k] < Jt.n_ids ? Jt.id2str[ia[k]] : NULL;
            if ((a == NULL) != (b == NULL) || (a && strcmp(a, b))) { fails++; printf("FAIL: piece %d decodes differently (%s vs %s)\n", ia[k], a ? a : "(null)", b ? b : "(null)"); break; }
        }
        lines++; toks += na;
    }
    printf("%d lines, %d ids, %d control tokens compared\n", lines, toks, Gt.nsp);
    if (fails) { printf("%d failure(s)\n", fails); return 1; }
    puts("all passed");
    return 0;
}
