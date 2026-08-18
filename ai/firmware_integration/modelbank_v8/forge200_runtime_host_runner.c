#include "forge200_runtime_v8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4U) << 32);
}

static uint8_t *read_file(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    uint8_t *data;
    long length;
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file); return NULL;
    }
    data = (uint8_t *)malloc((size_t)length);
    if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data); fclose(file); return NULL;
    }
    fclose(file); *bytes = (size_t)length; return data;
}

int main(int argc, char **argv)
{
    uint8_t *package, *golden;
    size_t package_bytes, golden_bytes;
    uint64_t payload_bytes;
    f2rt_model_t model;
    float *workspace;
    int result = 1;
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.icmf golden.f2gv\n", argv[0]); return 2;
    }
    package = read_file(argv[1], &package_bytes);
    golden = read_file(argv[2], &golden_bytes);
    if (package == NULL || golden == NULL || package_bytes < 256U || golden_bytes < 64U ||
        memcmp(package, "ICMF", 4U) != 0 || memcmp(golden, "F2GV", 4U) != 0) {
        fprintf(stderr, "FILE_OR_MAGIC_GATE\n"); goto done;
    }
    payload_bytes = rd64(package + 24U);
    if (payload_bytes > UINT32_MAX || payload_bytes > (uint64_t)SIZE_MAX - 256U ||
        package_bytes - 256U != (size_t)payload_bytes) {
        fprintf(stderr, "OUTER_BOUNDS_GATE:%llu:%llu\n", (unsigned long long)package_bytes, (unsigned long long)payload_bytes); goto done;
    }
    {
        f2rt_status_t bind_status = f2rt_bind(package + 256U, (uint32_t)payload_bytes, &model);
        if (bind_status != F2RT_OK) { fprintf(stderr, "BIND_GATE:%d\n", (int)bind_status); goto done; }
    }
    workspace = (float *)calloc(model.workspace_elems, sizeof(float));
    if (workspace == NULL) goto done;
    if (rd32(golden + 20U) == 4U) {
        uint32_t input_count = rd32(golden + 28U), output_count = rd32(golden + 32U), prompt_length = rd32(golden + 36U);
        const uint16_t *input;
        const uint16_t *expected;
        uint16_t *actual = NULL;
        uint32_t i;
        f2rt_status_t inference_status = F2RT_ERR_ARGUMENT;
        if (output_count != 0U && prompt_length != 0U && prompt_length <= input_count &&
            output_count <= SIZE_MAX / sizeof(uint16_t) &&
            f2rt_golden_layout_ok((uint64_t)golden_bytes, input_count, output_count,
                                  sizeof(uint16_t))) {
            input = (const uint16_t *)(const void *)(golden + 64U);
            expected = input + input_count;
            actual = (uint16_t *)calloc(output_count, sizeof(uint16_t));
        }
        if (actual != NULL &&
            (inference_status = f2rt_generate_u16(&model, input, prompt_length, actual, output_count, workspace, model.workspace_elems)) == F2RT_OK) {
            result = 0;
            for (i = 0U; i < output_count; ++i) if (actual[i] != expected[i]) { result = 1; break; }
        }
        if (result != 0) fprintf(stderr, "NANOLM_GATE:%d:%u:%u:%u\n", (int)inference_status, input_count, output_count, prompt_length);
        free(actual);
        if (result == 0) printf("{\"status\":\"PASS\",\"kind\":\"NANOLM\",\"outputs\":%u}\n", output_count);
    } else {
        uint32_t input_count = rd32(golden + 28U), output_count = rd32(golden + 32U), tolerance_bits = rd32(golden + 40U);
        const float *input;
        const float *expected;
        float *actual = NULL;
        float tolerance, maximum = 0.0f;
        uint32_t i;
        memcpy(&tolerance, &tolerance_bits, sizeof(tolerance));
        f2rt_status_t inference_status = F2RT_ERR_ARGUMENT;
        if (output_count != 0U && output_count <= SIZE_MAX / sizeof(float) &&
            f2rt_golden_layout_ok((uint64_t)golden_bytes, input_count, output_count,
                                  sizeof(float))) {
            input = (const float *)(const void *)(golden + 64U);
            expected = input + input_count;
            actual = (float *)calloc(output_count, sizeof(float));
        }
        if (actual != NULL &&
            (inference_status = f2rt_infer_f32(&model, input, input_count, actual, output_count, workspace, model.workspace_elems)) == F2RT_OK) {
            result = 0;
            for (i = 0U; i < output_count; ++i) {
                float difference = fabsf(actual[i] - expected[i]);
                if (difference > maximum) maximum = difference;
                if (!isfinite(actual[i]) || difference > tolerance) result = 1;
            }
        }
        if (result != 0) fprintf(stderr, "F32_GATE:%d:%u:%u:%.9g\n", (int)inference_status, input_count, output_count, maximum);
        free(actual);
        if (result == 0) printf("{\"status\":\"PASS\",\"kind\":\"F32\",\"outputs\":%u,\"max_abs_error\":%.9g}\n", output_count, maximum);
    }
    free(workspace);
done:
    free(package); free(golden);
    return result;
}
