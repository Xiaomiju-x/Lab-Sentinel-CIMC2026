/******************************************************************************
 * sha256.h — FIPS 180-4 SHA-256 (self-contained, no libs).
 * Used to hash-chain electronic batch records on-chip (tamper-evident
 * traceability, the 21 CFR Part 11 / AMS2750 electronic-record requirement).
 * Verified against NIST known-answer vectors in host_test/record_test.c.
 ******************************************************************************/
#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buflen;
} sha256_ctx;

void sha256_init(sha256_ctx *c);
void sha256_update(sha256_ctx *c, const void *data, size_t len);
void sha256_final(sha256_ctx *c, uint8_t out[32]);
/* one-shot convenience */
void sha256(const void *data, size_t len, uint8_t out[32]);
/* lowercase hex (65 bytes incl. NUL) */
void sha256_hex(const uint8_t hash[32], char out[65]);

#ifdef __cplusplus
}
#endif
#endif /* SHA256_H */
