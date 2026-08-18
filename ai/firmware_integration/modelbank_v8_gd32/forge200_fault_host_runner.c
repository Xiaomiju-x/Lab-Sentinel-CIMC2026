#include "forge200_modelbank.h"
#include "sha256.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *package;
    const char *golden_path;
} host_context_t;

static int read_at(void *context, uint64_t offset, void *destination,
                   uint32_t bytes)
{
    FILE *file = (FILE *)context;
    if (file == NULL || _fseeki64(file, (__int64)offset, SEEK_SET) != 0 ||
        fread(destination, 1, bytes, file) != bytes) {
        return -1;
    }
    return 0;
}

static int hash_at(void *context, uint64_t offset, uint64_t bytes,
                   uint8_t output[32])
{
    FILE *file = (FILE *)context;
    uint8_t buffer[8192];
    sha256_ctx sha;
    if (file == NULL || _fseeki64(file, (__int64)offset, SEEK_SET) != 0) {
        return -1;
    }
    sha256_init(&sha);
    while (bytes != 0U) {
        size_t chunk = bytes > sizeof(buffer) ? sizeof(buffer) : (size_t)bytes;
        if (fread(buffer, 1, chunk, file) != chunk) return -2;
        sha256_update(&sha, buffer, chunk);
        bytes -= chunk;
    }
    sha256_final(&sha, output);
    return 0;
}

static int engine_supported(void *context, uint16_t engine, uint16_t opset)
{
    (void)context;
    return (engine == 1U || engine == 2U || engine == 5U) && opset == 1U;
}

static int golden_check(void *context,
                        const forge200_package_info_t *package,
                        const uint8_t *payload, uint64_t payload_bytes)
{
    host_context_t *host = (host_context_t *)context;
    FILE *file;
    uint8_t digest[32];
    uint8_t buffer[8192];
    size_t got;
    sha256_ctx sha;
    (void)payload;
    (void)payload_bytes;
    file = fopen(host->golden_path, "rb");
    if (file == NULL) return 0;
    sha256_init(&sha);
    while ((got = fread(buffer, 1, sizeof(buffer), file)) != 0U) {
        sha256_update(&sha, buffer, got);
    }
    fclose(file);
    sha256_final(&sha, digest);
    return memcmp(digest, package->golden_sha256, 32U) == 0;
}

static int activate(void *context,
                    const forge200_package_info_t *package,
                    const uint8_t *payload, uint64_t payload_bytes)
{
    (void)context;
    (void)package;
    (void)payload;
    (void)payload_bytes;
    return 1;
}

int main(int argc, char **argv)
{
    FILE *package;
    __int64 package_bytes;
    uint8_t *slot_a;
    uint8_t *slot_b;
    forge200_reader_t reader;
    forge200_runtime_t runtime;
    forge200_modelbank_t bank;
    host_context_t context;
    forge200_status_t status;
    unsigned expected;
    if (argc != 6) {
        fprintf(stderr,
                "usage: runner package golden model catalog_generation expected_status\n");
        return 2;
    }
    package = fopen(argv[1], "rb");
    if (package == NULL || _fseeki64(package, 0, SEEK_END) != 0 ||
        (package_bytes = _ftelli64(package)) < 0 ||
        _fseeki64(package, 0, SEEK_SET) != 0) {
        return 3;
    }
    slot_a = (uint8_t *)malloc(0x740000U);
    slot_b = (uint8_t *)malloc(0x740000U);
    if (slot_a == NULL || slot_b == NULL) return 4;
    memset(&reader, 0, sizeof(reader));
    reader.context = package;
    reader.package_bytes = (uint64_t)package_bytes;
    reader.read = read_at;
    reader.sha256 = hash_at;
    memset(&context, 0, sizeof(context));
    context.package = package;
    context.golden_path = argv[2];
    memset(&runtime, 0, sizeof(runtime));
    runtime.context = &context;
    runtime.engine_supported = engine_supported;
    runtime.golden_check = golden_check;
    runtime.activate = activate;
    forge200_modelbank_init(&bank, slot_a, 0x740000U, slot_b, 0x740000U, 1U);
    status = forge200_modelbank_load(
        &bank, &reader, &runtime, argv[3],
        (uint64_t)_strtoui64(argv[4], NULL, 10));
    expected = (unsigned)strtoul(argv[5], NULL, 10);
    printf("{\"status\":%u,\"expected\":%u,\"pass\":%s}\n",
           (unsigned)status, expected,
           (unsigned)status == expected ? "true" : "false");
    fclose(package);
    free(slot_a);
    free(slot_b);
    return (unsigned)status == expected ? 0 : 1;
}
