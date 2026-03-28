/*
 * smoke_test.c — minimal ORT session loader + zero-fill inference.
 * Usage: ./smoke_test <path/to/model.ort>
 * Exit 0 = session loaded and inference succeeded, non-zero = failure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

/* Convenience macro: check an OrtStatus*, print and bail on failure. */
#define ORT_CHECK(expr, label)                                              \
    do {                                                                    \
        OrtStatus *_s = (expr);                                             \
        if (_s) {                                                           \
            fprintf(stderr, "SMOKE FAIL: %s: %s\n",                        \
                    (label), api->GetErrorMessage(_s));                     \
            api->ReleaseStatus(_s);                                         \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.ort>\n", argv[0]);
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

    OrtEnv            *env      = NULL;
    OrtSessionOptions *opts     = NULL;
    OrtSession        *session  = NULL;
    OrtMemoryInfo     *mem_info = NULL;

    /* Arrays sized after we know the input count. */
    size_t        input_count   = 0;
    char        **input_names   = NULL;  /* freed via AllocatorFree */
    OrtValue    **input_tensors = NULL;
    void        **input_bufs    = NULL;  /* zero-fill buffers; must outlive Run */
    OrtAllocator *allocator     = NULL;

    size_t        output_count   = 0;
    char        **output_names   = NULL;
    OrtValue    **output_tensors = NULL;

    /* ------------------------------------------------------------------ */
    /* 1. Env                                                               */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "smoke_test", &env),
              "CreateEnv");

    /* ------------------------------------------------------------------ */
    /* 2. Session options + session                                         */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->CreateSessionOptions(&opts), "CreateSessionOptions");
    ORT_CHECK(api->CreateSession(env, model_path, opts, &session),
              "CreateSession");

    fprintf(stdout, "SMOKE OK: loaded %s\n", model_path);
    fflush(stdout);

    /* ------------------------------------------------------------------ */
    /* 3. CPU memory info (needed to create tensors)                        */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                       &mem_info),
              "CreateCpuMemoryInfo");

    /* ------------------------------------------------------------------ */
    /* 4. Default allocator (used to query names)                           */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->GetAllocatorWithDefaultOptions(&allocator),
              "GetAllocatorWithDefaultOptions");

    /* ------------------------------------------------------------------ */
    /* 5. Query inputs and build zero-fill tensors                          */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->SessionGetInputCount(session, &input_count),
              "SessionGetInputCount");

    input_names   = calloc(input_count, sizeof(char *));
    input_tensors = calloc(input_count, sizeof(OrtValue *));
    input_bufs    = calloc(input_count, sizeof(void *));
    if (!input_names || !input_tensors || !input_bufs) {
        fprintf(stderr, "SMOKE FAIL: out of memory\n");
        goto cleanup;
    }

    for (size_t i = 0; i < input_count; i++) {
        /* Get input name (caller must free via allocator). */
        ORT_CHECK(api->SessionGetInputName(session, i, allocator,
                                           &input_names[i]),
                  "SessionGetInputName");

        /* Get type/shape info. */
        OrtTypeInfo *type_info = NULL;
        ORT_CHECK(api->SessionGetInputTypeInfo(session, i, &type_info),
                  "SessionGetInputTypeInfo");

        const OrtTensorTypeAndShapeInfo *shape_info = NULL;
        ORT_CHECK(api->CastTypeInfoToTensorInfo(type_info, &shape_info),
                  "CastTypeInfoToTensorInfo");

        ONNXTensorElementDataType elem_type;
        ORT_CHECK(api->GetTensorElementType(shape_info, &elem_type),
                  "GetTensorElementType");

        size_t ndim = 0;
        ORT_CHECK(api->GetDimensionsCount(shape_info, &ndim),
                  "GetDimensionsCount");

        int64_t *dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
        if (!dims) {
            api->ReleaseTypeInfo(type_info);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        ORT_CHECK(api->GetDimensions(shape_info, dims, ndim),
                  "GetDimensions");
        api->ReleaseTypeInfo(type_info);

        /* Replace dynamic dimensions (≤ 0) with 1. */
        size_t total_elems = 1;
        for (size_t d = 0; d < ndim; d++) {
            if (dims[d] <= 0) dims[d] = 1;
            total_elems *= (size_t)dims[d];
        }

        /* Determine element size. */
        size_t elem_size;
        switch (elem_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: elem_size = 2; break;
            default:                                    elem_size = 4; break;
        }

        /* Allocate zero-filled buffer; kept alive until after Run(). */
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

    /* ------------------------------------------------------------------ */
    /* 6. Query output names                                                */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->SessionGetOutputCount(session, &output_count),
              "SessionGetOutputCount");

    output_names   = calloc(output_count, sizeof(char *));
    output_tensors = calloc(output_count, sizeof(OrtValue *));
    if (!output_names || !output_tensors) {
        fprintf(stderr, "SMOKE FAIL: out of memory\n");
        goto cleanup;
    }

    for (size_t i = 0; i < output_count; i++) {
        ORT_CHECK(api->SessionGetOutputName(session, i, allocator,
                                            &output_names[i]),
                  "SessionGetOutputName");
    }

    /* ------------------------------------------------------------------ */
    /* 7. Run inference                                                     */
    /* ------------------------------------------------------------------ */
    ORT_CHECK(api->Run(session, NULL,
                       (const char *const *)input_names,  input_tensors,  input_count,
                       (const char *const *)output_names, output_tensors, output_count),
              "Run");

    fprintf(stdout, "SMOKE OK: inference succeeded (%zu input(s), %zu output(s))\n",
            input_count, output_count);
    fflush(stdout);
    exit_code = 0;

cleanup:
    /* Release output tensors and names. */
    if (output_tensors) {
        for (size_t i = 0; i < output_count; i++)
            if (output_tensors[i]) api->ReleaseValue(output_tensors[i]);
        free(output_tensors);
    }
    if (output_names) {
        for (size_t i = 0; i < output_count; i++)
            if (output_names[i])
                api->AllocatorFree(allocator, output_names[i]);
        free(output_names);
    }
    /* Release input tensors BEFORE freeing the backing buffers. */
    if (input_tensors) {
        for (size_t i = 0; i < input_count; i++)
            if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
        free(input_tensors);
    }
    if (input_bufs) {
        for (size_t i = 0; i < input_count; i++)
            free(input_bufs[i]);
        free(input_bufs);
    }
    if (input_names) {
        for (size_t i = 0; i < input_count; i++)
            if (input_names[i])
                api->AllocatorFree(allocator, input_names[i]);
        free(input_names);
    }
    if (mem_info) api->ReleaseMemoryInfo(mem_info);
    if (session)  api->ReleaseSession(session);
    if (opts)     api->ReleaseSessionOptions(opts);
    if (env)      api->ReleaseEnv(env);
    return exit_code;
}
