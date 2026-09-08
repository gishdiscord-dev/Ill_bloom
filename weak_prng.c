#include "weak_prng.h"
#include "wordlist.h"

#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define WL_SIZE BIP39_WORDLIST_SIZE

static int unpack_11bit(const uint8_t *bits, int bit_offset)
{
    int idx = 0;
    for (int b = 0; b < 11; b++) {
        int gbit = bit_offset + b;
        int bpos = gbit / 8;
        int bshf = 7 - (gbit % 8);
        if (bits[bpos] & (1 << bshf))
            idx |= (1 << (10 - b));
    }
    return idx;
}

static int bip39_checksum(const uint8_t *entropy, int ent_bytes, int cs_len)
{
    uint8_t hash[SHA256_DIGEST_LENGTH];
    SHA256(entropy, ent_bytes, hash);
    int cs = 0;
    for (int i = 0; i < cs_len; i++) {
        if (hash[0] & (1 << (7 - i)))
            cs |= (1 << (cs_len - 1 - i));
    }
    return cs;
}

static void indices_to_string(const int *indices, int count,
                              char *output, size_t outlen)
{
    if (!output || outlen == 0)
        return;
    output[0] = '\0';

    for (int i = 0; i < count; i++) {
        const char *w = BIP39_WORDLIST[indices[i]];
        size_t len = strlen(output);
        size_t wlen = strlen(w);
        size_t need = wlen + (i ? 1u : 0u);
        if (len + need + 1 > outlen)
            return;
        if (i)
            output[len++] = ' ';
        memcpy(output + len, w, wlen + 1);
    }
}

int weak_prng_init_trace(weak_prng_ctx_t *ctx,
                         double initial_math_random,
                         const uint8_t *sign_bits,
                         size_t sign_count)
{
    if (!ctx || !sign_bits || sign_count > sizeof(ctx->sign_storage))
        return -1;
    memset(ctx, 0, sizeof(*ctx));
    memcpy(ctx->sign_storage, sign_bits, sign_count);
    cryptojs_trace_init(&ctx->trace, initial_math_random,
                        ctx->sign_storage, sign_count);
    ctx->trace_ready = 1;
    return 0;
}

int weak_prng_generate_entropy(weak_prng_ctx_t *ctx,
                               uint8_t *entropy,
                               size_t ent_bytes)
{
    if (!ctx || !ctx->trace_ready || !entropy)
        return -2;
    if (ent_bytes != 16 && ent_bytes != 32)
        return -1;
    return cryptojs_wordarray_random_trace(&ctx->trace, entropy, ent_bytes);
}

int weak_prng_generate_mnemonic(weak_prng_ctx_t *ctx,
                                char *output,
                                size_t outlen)
{
    if (!ctx || !ctx->trace_ready || !output || outlen == 0)
        return -2;

    uint8_t ent[16];
    if (weak_prng_generate_entropy(ctx, ent, sizeof(ent)) != 0)
        return -1;

    const int cs_bits = 4;
    const int word_count = 12;
    uint8_t bits[17] = {0};
    memcpy(bits, ent, sizeof(ent));

    int cs = bip39_checksum(ent, sizeof(ent), cs_bits);
    for (int i = 0; i < cs_bits; i++) {
        if (cs & (1 << (cs_bits - 1 - i))) {
            int bp = 128 + i;
            bits[bp / 8] |= (1 << (7 - (bp % 8)));
        }
    }

    int indices[12];
    for (int i = 0; i < word_count; i++)
        indices[i] = unpack_11bit(bits, i * 11);

    indices_to_string(indices, word_count, output, outlen);
    return word_count;
}

/* ---------- disabled legacy candidate-generation interface ---------- */

void weak_prng_init(weak_prng_ctx_t *ctx)
{
    if (ctx) memset(ctx, 0, sizeof(*ctx));
}

void weak_prng_init_seeded(weak_prng_ctx_t *ctx, uint32_t seed_base)
{
    (void)seed_base;
    weak_prng_init(ctx);
}

void weak_prng_init_biased(weak_prng_ctx_t *ctx, uint32_t seed_base, int bias_level)
{
    (void)seed_base; (void)bias_level;
    weak_prng_init(ctx);
}

uint32_t weak_prng_next(weak_prng_ctx_t *ctx)
{
    (void)ctx;
    return 0;
}

int weak_prng_word_index(weak_prng_ctx_t *ctx, int wordlist_size)
{
    (void)ctx; (void)wordlist_size;
    return -1;
}

int weak_prng_generate_illbloom_pattern(weak_prng_ctx_t *ctx, char *output,
                                        size_t outlen)
{
    /* Ill Bloom does not impose a mnemonic prefix. */
    return weak_prng_generate_mnemonic(ctx, output, outlen);
}

int weak_prng_generate_batch(weak_prng_ctx_t *ctx,
                             char (*outputs)[WEAK_PRNG_MAX_SEED_LEN],
                             int count, int word_count)
{
    (void)ctx; (void)outputs; (void)count; (void)word_count;
    return 0;
}

float weak_prng_score_mnemonic(const char *mnemonic,
                               const char *wordlist[], int wordlist_size)
{
    (void)mnemonic; (void)wordlist; (void)wordlist_size;
    /* A mnemonic's visible word-index distribution does not establish that
     * vulnerable CryptoJS generated it. */
    return 0.0f;
}

int weak_prng_estimate_entropy(const char *mnemonic,
                               const char *wordlist[], int wordlist_size)
{
    (void)mnemonic; (void)wordlist; (void)wordlist_size;
    /* Nominal entropy for a 12-word BIP39 phrase; effective generator search
     * space is a property of the generation mechanism, not inferable from the
     * phrase alone. */
    return 128;
}
