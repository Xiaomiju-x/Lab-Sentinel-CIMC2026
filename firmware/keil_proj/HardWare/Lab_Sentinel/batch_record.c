/******************************************************************************
 * batch_record.c — tamper-evident hash-chained batch records. See header.
 ******************************************************************************/
#include "batch_record.h"
#include "sha256.h"
#include <string.h>

const uint8_t BR_GENESIS_PREV[32] = { 0 };

/* ---- explicit big-endian serialization (padding-independent) ---- */
static void put_u8(uint8_t **p, uint8_t v)  { *(*p)++ = v; }
static void put_u16(uint8_t **p, uint16_t v){ put_u8(p,(uint8_t)(v>>8)); put_u8(p,(uint8_t)v); }
static void put_u32(uint8_t **p, uint32_t v){ put_u16(p,(uint16_t)(v>>16)); put_u16(p,(uint16_t)v); }
static void put_f32(uint8_t **p, float f)
{
    uint32_t u; memcpy(&u, &f, 4); put_u32(p, u);   /* IEEE-754 bit pattern, BE */
}
static void put_bytes(uint8_t **p, const void *src, int n)
{
    memcpy(*p, src, (size_t)n); *p += n;
}

/* serialize everything EXCEPT this_hash (prev_hash IS included). returns length */
static int serialize(const batch_record_t *r, uint8_t *buf)
{
    uint8_t *p = buf;
    put_u32(&p, r->batch_id);
    put_u32(&p, r->unix_time);
    put_bytes(&p, r->recipe, BR_RECIPE_LEN);
    put_bytes(&p, r->operator_id, BR_OPERATOR_LEN);
    put_f32(&p, r->peak_C);
    put_f32(&p, r->soak_cpk);
    put_f32(&p, r->elem_remaining_pct);
    put_u16(&p, r->n_ai_alarms);
    put_u8(&p, r->in_control);
    put_u8(&p, r->capable);
    put_u8(&p, r->final_state);
    put_u8(&p, r->fault);
    put_bytes(&p, r->prev_hash, 32);
    return (int)(p - buf);
}

static void compute_hash(const batch_record_t *r, uint8_t out[32])
{
    uint8_t buf[128];
    int n = serialize(r, buf);
    sha256(buf, (size_t)n, out);
}

void batch_record_seal(batch_record_t *r, const uint8_t prev_hash[32])
{
    memcpy(r->prev_hash, prev_hash, 32);
    compute_hash(r, r->this_hash);
}

int batch_record_verify(const batch_record_t *r, const uint8_t prev_hash[32])
{
    uint8_t h[32];
    if (memcmp(r->prev_hash, prev_hash, 32) != 0) return 0;
    compute_hash(r, h);
    return (memcmp(h, r->this_hash, 32) == 0) ? 1 : 0;
}

int batch_chain_verify(const batch_record_t *recs, int n)
{
    const uint8_t *prev = BR_GENESIS_PREV;
    int i;
    for (i = 0; i < n; i++) {
        if (!batch_record_verify(&recs[i], prev)) return i;
        prev = recs[i].this_hash;
    }
    return -1;
}
