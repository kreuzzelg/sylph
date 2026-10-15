/* sylph (2026-10-14): the Qwen pre-tokenizer's Unicode classes. The hand-written
 * table in qwen36.c ended at U+024F for Latin letters and knew no \p{N} beyond the
 * ASCII/Arabic/fullwidth digits, so "həd" (schwa, U+0259), "Cẩm" (U+1EA9) and
 * "4 ½" (U+00BD, category No) split differently from llama.cpp -- found by the
 * equivalence harness's tokenizer gate on wikitext-2 (5 pieces in 297 193 ids).
 * unicode_cat.h now carries the general-category ranges; this pins the classes and
 * the piece boundaries the regex implies, with no model file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int g_fails = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* split `s` with pretok_end and compare the pieces */
static void expect_pieces(const char *s, const char **want, int nwant) {
    int n = (int)strlen(s), i = 0, k = 0, ok = 1;
    while (i < n && k < nwant) {
        int e = pretok_end(s, i, n);
        if (e <= i) { ok = 0; break; }
        if ((int)strlen(want[k]) != e - i || strncmp(s + i, want[k], (size_t)(e - i))) { ok = 0; break; }
        i = e; k++;
    }
    if (ok && (i != n || k != nwant)) ok = 0;
    if (!ok) {
        g_fails++; printf("FAIL pieces of %s: got [", s);
        for (int j = 0; j < n;) { int e = pretok_end(s, j, n); if (e <= j) e = j + 1; printf("%s'%.*s'", j ? " " : "", e - j, s + j); j = e; }
        printf("] want ["); for (int j = 0; j < nwant; j++) printf("%s'%s'", j ? " " : "", want[j]); printf("]\n");
    }
}
#define PIECES(s, ...) do { const char *w[] = {__VA_ARGS__}; expect_pieces(s, w, (int)(sizeof w / sizeof w[0])); } while (0)

int main(void) {
    g_tok_llama = 1;   /* the classes of a GGUF tokenizer (load_tokenizer_gguf sets this); the container keeps upstream's tables */
    /* classes */
    CHECK(uclass('a') == U_L && uclass('Z') == U_L && uclass('7') == U_N && uclass(' ') == U_W && uclass('.') == U_P, "ASCII classes");
    CHECK(uclass(0x0259) == U_L, "U+0259 schwa is a letter");
    CHECK(uclass(0x1EA9) == U_L, "U+1EA9 (Vietnamese) is a letter");
    CHECK(uclass(0x00E9) == U_L && uclass(0x0416) == U_L && uclass(0x4E2D) == U_L && uclass(0xAC00) == U_L, "é Ж 中 가 are letters");
    CHECK(uclass(0x02B0) == U_L, "U+02B0 modifier letter (Lm) is a letter");
    CHECK(uclass(0x00BD) == U_N, "U+00BD ½ is a number (No)");
    CHECK(uclass(0x00B2) == U_N && uclass(0x2167) == U_N && uclass(0x0663) == U_N && uclass(0x2460) == U_N, "² Ⅷ ٣ ① are numbers");
    CHECK(uclass(0x0301) == U_M && uclass(0x093F) == U_M, "combining acute and devanagari vowel sign are marks");
    CHECK(uclass(0x0085) == U_W && uclass(0x00A0) == U_W && uclass(0x2003) == U_W && uclass(0x3000) == U_W && uclass(0x1680) == U_W, "White_Space");
    CHECK(uclass(0xFEFF) != U_W && uclass(0x200B) != U_W, "BOM and ZWSP are not White_Space");
    CHECK(uclass(0x2014) != U_L && uclass(0x2014) != U_N && uclass(0x2014) != U_W, "em dash is punctuation/other");
    CHECK(uclass(0x1F600) != U_L && uclass(0x1F600) != U_N, "emoji is not a letter or number");
    /* piece boundaries of the Qwen2 regex */
    PIECES(" h\xC9\x99" "d", " h\xC9\x99" "d");                                    /* " həd" one letter run */
    PIECES(" C\xE1\xBA\xA9m", " C\xE1\xBA\xA9m");                                   /* " Cẩm" */
    PIECES("4 \xC2\xBD", "4", " ", "\xC2\xBD");                                      /* "4 ½": digits stand alone */
    PIECES("12345", "1", "2", "3", "4", "5");
    PIECES("e\xCC\x81" "t\xC3\xA9", "e\xCC\x81" "t\xC3\xA9");                       /* combining mark continues the run */
    PIECES("x\xC2\xA0y", "x", "\xC2\xA0y");                                         /* NBSP is whitespace: " y"-style prefix */
    PIECES("a \xE2\x80\x94 b", "a", " \xE2\x80\x94", " b");                          /* " —" punctuation run with space */
    PIECES("\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0 \xE6\x9D\xB1\xE4\xBA\xAC", "\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0", " \xE6\x9D\xB1\xE4\xBA\xAC");
    if (g_fails) { printf("%d failure(s)\n", g_fails); return 1; }
    puts("test_unicode_cat: all passed");
    return 0;
}
