/*
 * smoke_test.c — minimal ORT session loader with two modes.
 *
 *   ./smoke_test <model.ort>
 *       Zero-fill inference; exit 0 if the session loads and Run() succeeds.
 *
 *   ./smoke_test <model.ort> <vectors.tvbin> <cosine_threshold>
 *       Replay tokenized inputs from the test-vectors file and compare the first
 *       output tensor to the stored reference by cosine similarity. Exit 0 only
 *       if every sample is >= threshold.
 *
 * Runtime graph optimization is disabled (ORT_DISABLE_ALL): the model is already
 * fully optimized offline, and re-optimizing at load time introduces fused ops
 * whose shape requirements conflict with broadcast attention bias.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

#define ORT_CHECK(expr, label)                                              \
    do {                                                                    \
        OrtStatus *_s = (expr);                                             \
        if (_s) {                                                           \
            fprintf(stderr, "SMOKE FAIL: %s: %s\n",                         \
                    (label), api->GetErrorMessage(_s));                     \
            api->ReleaseStatus(_s);                                         \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

static ONNXTensorElementDataType code_to_ort(uint32_t code) {
    switch (code) {
        case 0: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case 1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case 2: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case 3: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

/* Convert the first output tensor to a newly-allocated float array. */
static float *tensor_to_float(const OrtApi *api, OrtValue *val, size_t *out_count) {
    OrtTensorTypeAndShapeInfo *info = NULL;
    if (api->GetTensorTypeAndShape(val, &info)) return NULL;
    size_t count = 0;
    api->GetTensorShapeElementCount(info, &count);
    ONNXTensorElementDataType t;
    api->GetTensorElementType(info, &t);
    api->ReleaseTensorTypeAndShapeInfo(info);

    void *data = NULL;
    if (api->GetTensorMutableData(val, &data)) return NULL;

    float *out = malloc((count ? count : 1) * sizeof(float));
    if (!out) return NULL;
    for (size_t i = 0; i < count; i++) {
        switch (t) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                out[i] = ((float *)data)[i]; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                out[i] = half_to_float(((uint16_t *)data)[i]); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                out[i] = (float)((double *)data)[i]; break;
            default:
                out[i] = 0.0f; break;
        }
    }
    *out_count = count;
    return out;
}

static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static int read_exact(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

/* Comparison mode: replay tokenized inputs, compare output[0] by cosine. */
static int run_comparison(const OrtApi *api, OrtSession *session,
                          OrtMemoryInfo *mem_info, OrtAllocator *allocator,
                          const char *tvpath, double threshold) {
    int rc = 1;
    char *out_name = NULL;
    FILE *f = fopen(tvpath, "rb");
    if (!f) {
        fprintf(stderr, "SMOKE FAIL: cannot open %s\n", tvpath);
        return 1;
    }

    char magic[4];
    uint32_t num_samples = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "TVB1", 4) != 0 ||
        !read_exact(f, &num_samples, 4)) {
        fprintf(stderr, "SMOKE FAIL: bad or truncated test-vectors header\n");
        fclose(f);
        return 1;
    }

    if (api->SessionGetOutputName(session, 0, allocator, &out_name)) {
        fprintf(stderr, "SMOKE FAIL: SessionGetOutputName failed\n");
        fclose(f);
        return 1;
    }

    int all_ok = 1;
    for (uint32_t si = 0; si < num_samples; si++) {
        uint32_t num_inputs = 0;
        if (!read_exact(f, &num_inputs, 4)) {
            fprintf(stderr, "SMOKE FAIL: truncated sample %u\n", si);
            all_ok = 0;
            break;
        }

        char **names = calloc(num_inputs, sizeof(char *));
        OrtValue **tensors = calloc(num_inputs, sizeof(OrtValue *));
        void **bufs = calloc(num_inputs, sizeof(void *));
        int sample_bad = (!names || !tensors || !bufs);

        for (uint32_t ii = 0; ii < num_inputs && !sample_bad; ii++) {
            uint32_t name_len = 0, dcode = 0, ndim = 0;
            if (!read_exact(f, &name_len, 4)) { sample_bad = 1; break; }
            names[ii] = malloc(name_len + 1);
            if (!names[ii] || !read_exact(f, names[ii], name_len)) { sample_bad = 1; break; }
            names[ii][name_len] = '\0';
            if (!read_exact(f, &dcode, 4) || !read_exact(f, &ndim, 4)) { sample_bad = 1; break; }
            int64_t *dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
            for (uint32_t d = 0; d < ndim; d++) {
                if (!read_exact(f, &dims[d], 8)) { sample_bad = 1; break; }
            }
            uint64_t nbytes = 0;
            if (sample_bad || !read_exact(f, &nbytes, 8)) { free(dims); sample_bad = 1; break; }
            bufs[ii] = malloc(nbytes ? nbytes : 1);
            if (!bufs[ii] || !read_exact(f, bufs[ii], nbytes)) { free(dims); sample_bad = 1; break; }
            OrtStatus *ts = api->CreateTensorWithDataAsOrtValue(
                mem_info, bufs[ii], nbytes, dims, ndim, code_to_ort(dcode), &tensors[ii]);
            free(dims);
            if (ts) {
                fprintf(stderr, "SMOKE FAIL: CreateTensor: %s\n", api->GetErrorMessage(ts));
                api->ReleaseStatus(ts);
                sample_bad = 1;
                break;
            }
        }

        uint64_t ref_count = 0;
        float *ref = NULL;
        if (!sample_bad && read_exact(f, &ref_count, 8)) {
            ref = malloc((ref_count ? ref_count : 1) * sizeof(float));
            if (!ref || !read_exact(f, ref, ref_count * sizeof(float))) sample_bad = 1;
        } else {
            sample_bad = 1;
        }

        OrtValue *output = NULL;
        if (!sample_bad) {
            OrtStatus *rs = api->Run(session, NULL,
                (const char *const *)names, (const OrtValue *const *)tensors, num_inputs,
                (const char *const *)&out_name, 1, &output);
            if (rs) {
                fprintf(stderr, "SMOKE FAIL: Run: %s\n", api->GetErrorMessage(rs));
                api->ReleaseStatus(rs);
                sample_bad = 1;
            }
        }

        if (!sample_bad && output) {
            size_t got = 0;
            float *ov = tensor_to_float(api, output, &got);
            if (!ov) {
                sample_bad = 1;
            } else if (got != (size_t)ref_count) {
                fprintf(stderr, "SMOKE FAIL: sample %u output count %zu != ref %llu\n",
                        si, got, (unsigned long long)ref_count);
                sample_bad = 1;
            } else {
                double sim = cosine(ov, ref, got);
                printf("  sample %u: cosine=%.6f (threshold %.6f)\n", si, sim, threshold);
                if (sim < threshold) all_ok = 0;
            }
            free(ov);
        }
        if (sample_bad) all_ok = 0;

        if (output) api->ReleaseValue(output);
        for (uint32_t ii = 0; ii < num_inputs; ii++) {
            if (tensors && tensors[ii]) api->ReleaseValue(tensors[ii]);
            if (bufs) free(bufs[ii]);
            if (names) free(names[ii]);
        }
        free(tensors);
        free(bufs);
        free(names);
        free(ref);
        if (sample_bad) break;
    }

    if (all_ok) {
        printf("SMOKE OK: all %u sample(s) within cosine threshold %.6f\n",
               num_samples, threshold);
        rc = 0;
    } else {
        fprintf(stderr, "SMOKE FAIL: one or more samples below cosine threshold\n");
    }

    api->AllocatorFree(allocator, out_name);
    fclose(f);
    return rc;
}

/* Zero-fill mode: verify the session runs on all-zero inputs. */
static int run_zerofill(const OrtApi *api, OrtSession *session,
                        OrtMemoryInfo *mem_info, OrtAllocator *allocator) {
    int exit_code = 1;
    size_t input_count = 0;
    char **input_names = NULL;
    OrtValue **input_tensors = NULL;
    void **input_bufs = NULL;
    int64_t input_ids_seq_len = 1;
    size_t output_count = 0;
    char **output_names = NULL;
    OrtValue **output_tensors = NULL;

    ORT_CHECK(api->SessionGetInputCount(session, &input_count), "SessionGetInputCount");
    input_names = calloc(input_count, sizeof(char *));
    input_tensors = calloc(input_count, sizeof(OrtValue *));
    input_bufs = calloc(input_count, sizeof(void *));
    if (!input_names || !input_tensors || !input_bufs) {
        fprintf(stderr, "SMOKE FAIL: out of memory\n");
        goto cleanup;
    }

    for (size_t i = 0; i < input_count; i++) {
        char *name = NULL;
        OrtTypeInfo *type_info = NULL;
        const OrtTensorTypeAndShapeInfo *shape_info = NULL;
        size_t ndim = 0;
        int64_t *dims = NULL;
        ORT_CHECK(api->SessionGetInputName(session, i, allocator, &name), "SessionGetInputName");
        ORT_CHECK(api->SessionGetInputTypeInfo(session, i, &type_info), "SessionGetInputTypeInfo");
        ORT_CHECK(api->CastTypeInfoToTensorInfo(type_info, &shape_info), "CastTypeInfoToTensorInfo");
        ORT_CHECK(api->GetDimensionsCount(shape_info, &ndim), "GetDimensionsCount");
        dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
        if (!dims) {
            api->AllocatorFree(allocator, name);
            api->ReleaseTypeInfo(type_info);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        ORT_CHECK(api->GetDimensions(shape_info, dims, ndim), "GetDimensions");
        if (strcmp(name, "input_ids") == 0 && ndim >= 2 && dims[1] > 0) {
            input_ids_seq_len = dims[1];
        }
        free(dims);
        api->AllocatorFree(allocator, name);
        api->ReleaseTypeInfo(type_info);
    }

    for (size_t i = 0; i < input_count; i++) {
        ORT_CHECK(api->SessionGetInputName(session, i, allocator, &input_names[i]),
                  "SessionGetInputName");
        OrtTypeInfo *type_info = NULL;
        ORT_CHECK(api->SessionGetInputTypeInfo(session, i, &type_info), "SessionGetInputTypeInfo");
        const OrtTensorTypeAndShapeInfo *shape_info = NULL;
        ORT_CHECK(api->CastTypeInfoToTensorInfo(type_info, &shape_info), "CastTypeInfoToTensorInfo");
        ONNXTensorElementDataType elem_type;
        ORT_CHECK(api->GetTensorElementType(shape_info, &elem_type), "GetTensorElementType");
        size_t ndim = 0;
        ORT_CHECK(api->GetDimensionsCount(shape_info, &ndim), "GetDimensionsCount");
        int64_t *dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
        if (!dims) {
            api->ReleaseTypeInfo(type_info);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        ORT_CHECK(api->GetDimensions(shape_info, dims, ndim), "GetDimensions");
        api->ReleaseTypeInfo(type_info);

        size_t total_elems = 1;
        for (size_t d = 0; d < ndim; d++) {
            if (dims[d] <= 0) dims[d] = 1;
            if (strcmp(input_names[i], "attention_mask") == 0 && ndim >= 2 && d == 1) {
                dims[1] = input_ids_seq_len;
            }
            total_elems *= (size_t)dims[d];
        }

        size_t elem_size;
        switch (elem_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: elem_size = 2; break;
            default:                                    elem_size = 4; break;
        }

        input_bufs[i] = calloc(total_elems, elem_size);
        if (!input_bufs[i]) {
            free(dims);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        OrtStatus *ts = api->CreateTensorWithDataAsOrtValue(
                mem_info, input_bufs[i], total_elems * elem_size,
                dims, ndim, elem_type, &input_tensors[i]);
        free(dims);
        if (ts) {
            fprintf(stderr, "SMOKE FAIL: CreateTensorWithDataAsOrtValue: %s\n",
                    api->GetErrorMessage(ts));
            api->ReleaseStatus(ts);
            goto cleanup;
        }
        fprintf(stdout, "  input[%zu] = %s  (elem_type=%d, total_elems=%zu)\n",
                i, input_names[i], (int)elem_type, total_elems);
    }

    ORT_CHECK(api->SessionGetOutputCount(session, &output_count), "SessionGetOutputCount");
    output_names = calloc(output_count, sizeof(char *));
    output_tensors = calloc(output_count, sizeof(OrtValue *));
    if (!output_names || !output_tensors) {
        fprintf(stderr, "SMOKE FAIL: out of memory\n");
        goto cleanup;
    }
    for (size_t i = 0; i < output_count; i++) {
        ORT_CHECK(api->SessionGetOutputName(session, i, allocator, &output_names[i]),
                  "SessionGetOutputName");
    }

    ORT_CHECK(api->Run(session, NULL,
                       (const char *const *)input_names, input_tensors, input_count,
                       (const char *const *)output_names, output_count, output_tensors),
              "Run");
    fprintf(stdout, "SMOKE OK: inference succeeded (%zu input(s), %zu output(s))\n",
            input_count, output_count);
    exit_code = 0;

cleanup:
    if (output_tensors) {
        for (size_t i = 0; i < output_count; i++)
            if (output_tensors[i]) api->ReleaseValue(output_tensors[i]);
        free(output_tensors);
    }
    if (output_names) {
        for (size_t i = 0; i < output_count; i++)
            if (output_names[i]) api->AllocatorFree(allocator, output_names[i]);
        free(output_names);
    }
    if (input_tensors) {
        for (size_t i = 0; i < input_count; i++)
            if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
        free(input_tensors);
    }
    if (input_bufs) {
        for (size_t i = 0; i < input_count; i++) free(input_bufs[i]);
        free(input_bufs);
    }
    if (input_names) {
        for (size_t i = 0; i < input_count; i++)
            if (input_names[i]) api->AllocatorFree(allocator, input_names[i]);
        free(input_names);
    }
    return exit_code;
}

int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "Usage: %s <model.ort> [vectors.tvbin cosine_threshold]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    int exit_code = 1;

    const OrtApiBase *base = OrtGetApiBase();
    if (!base) {
        fprintf(stderr, "SMOKE FAIL: OrtGetApiBase() returned NULL\n");
        return 1;
    }
    const OrtApi *api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        fprintf(stderr, "SMOKE FAIL: could not get ORT API\n");
        return 1;
    }

    OrtEnv *env = NULL;
    OrtSessionOptions *opts = NULL;
    OrtSession *session = NULL;
    OrtMemoryInfo *mem_info = NULL;
    OrtAllocator *allocator = NULL;

    ORT_CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "smoke_test", &env), "CreateEnv");
    ORT_CHECK(api->CreateSessionOptions(&opts), "CreateSessionOptions");
    ORT_CHECK(api->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL),
              "SetSessionGraphOptimizationLevel");
    ORT_CHECK(api->CreateSession(env, model_path, opts, &session), "CreateSession");
    fprintf(stdout, "SMOKE OK: loaded %s\n", model_path);
    fflush(stdout);

    ORT_CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info),
              "CreateCpuMemoryInfo");
    ORT_CHECK(api->GetAllocatorWithDefaultOptions(&allocator),
              "GetAllocatorWithDefaultOptions");

    if (argc == 4) {
        exit_code = run_comparison(api, session, mem_info, allocator, argv[2], atof(argv[3]));
    } else {
        exit_code = run_zerofill(api, session, mem_info, allocator);
    }

cleanup:
    if (mem_info) api->ReleaseMemoryInfo(mem_info);
    if (session)  api->ReleaseSession(session);
    if (opts)     api->ReleaseSessionOptions(opts);
    if (env)      api->ReleaseEnv(env);
    return exit_code;
}
