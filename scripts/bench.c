/*
 * bench.c — minimal ORT inference benchmark harness.
 *
 *   ./bench <model.ort> <vectors.tvbin> <warmup> <iters>
 *       Load the model, replay the tokenized inputs from the test-vectors file
 *       in a timed loop, and emit latency/throughput/memory/size metrics as a
 *       single JSON object on stdout.
 *
 * Reuses the TVB1 reader and session setup from smoke_test.c. When the .tvbin
 * carries an fp32 golden reference (produced by gen_reference_vectors.py WITHOUT
 * --inputs-only), the harness additionally scores each output by cosine similarity
 * in an untimed pass and reports `mean_cosine`/`min_cosine` — purely informational,
 * never a gate. Vectors produced with `--inputs-only` carry an empty reference
 * payload, and the cosine fields are then omitted.
 *
 * Runtime graph optimization is disabled (ORT_DISABLE_ALL) to match the smoke
 * test and what ships: the model is already fully optimized offline. Inference
 * is single-threaded, batch=1 (SetIntraOpNumThreads(1)/SetInterOpNumThreads(1))
 * to model a serverless/Lambda profile.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include "onnxruntime_c_api.h"

#define ORT_CHECK(expr, label)                                              \
    do {                                                                    \
        OrtStatus *_s = (expr);                                             \
        if (_s) {                                                           \
            fprintf(stderr, "BENCH FAIL: %s: %s\n",                         \
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
        default:
            fprintf(stderr,
                    "BENCH WARN: unknown .tvbin dtype code %u; assuming float32\n",
                    code);
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
}

static int read_exact(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

/* IEEE half → float (mirrors smoke_test.c so fp16 outputs compare correctly). */
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

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Nearest-rank percentile over a sorted ascending array. */
static double percentile(const double *sorted, size_t n, double frac) {
    if (n == 0) return 0.0;
    double pos = frac * (double)(n - 1) + 0.5;
    size_t idx = (size_t)pos;
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* One decoded sample: named input tensors ready to feed to Run(), plus an
 * optional float32 reference embedding (ref_count == 0 under --inputs-only). */
typedef struct {
    uint32_t num_inputs;
    char **names;
    OrtValue **tensors;
    void **bufs;
    float *ref;
    size_t ref_count;
} Sample;

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <model.ort> <vectors.tvbin> <warmup> <iters>\n",
                argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *tvpath = argv[2];
    long warmup = atol(argv[3]);
    long iters = atol(argv[4]);
    if (iters <= 0) {
        fprintf(stderr, "BENCH FAIL: iters must be > 0\n");
        return 1;
    }
    if (warmup < 0) warmup = 0;

    int exit_code = 1;
    const OrtApiBase *base = OrtGetApiBase();
    if (!base) {
        fprintf(stderr, "BENCH FAIL: OrtGetApiBase() returned NULL\n");
        return 1;
    }
    const OrtApi *api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        fprintf(stderr, "BENCH FAIL: could not get ORT API\n");
        return 1;
    }

    OrtEnv *env = NULL;
    OrtSessionOptions *opts = NULL;
    OrtSession *session = NULL;
    OrtMemoryInfo *mem_info = NULL;
    OrtAllocator *allocator = NULL;
    char *out_name = NULL;
    Sample *samples = NULL;
    uint32_t num_samples = 0;
    double *times = NULL;
    FILE *f = NULL;

    ORT_CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env), "CreateEnv");
    ORT_CHECK(api->CreateSessionOptions(&opts), "CreateSessionOptions");
    ORT_CHECK(api->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL),
              "SetSessionGraphOptimizationLevel");
    ORT_CHECK(api->SetIntraOpNumThreads(opts, 1), "SetIntraOpNumThreads");
    ORT_CHECK(api->SetInterOpNumThreads(opts, 1), "SetInterOpNumThreads");

    double load_start = now_ms();
    ORT_CHECK(api->CreateSession(env, model_path, opts, &session), "CreateSession");
    double load_ms = now_ms() - load_start;

    ORT_CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info),
              "CreateCpuMemoryInfo");
    ORT_CHECK(api->GetAllocatorWithDefaultOptions(&allocator),
              "GetAllocatorWithDefaultOptions");
    ORT_CHECK(api->SessionGetOutputName(session, 0, allocator, &out_name),
              "SessionGetOutputName");

    /* Load all samples into memory (tensors persist across iterations). */
    f = fopen(tvpath, "rb");
    if (!f) {
        fprintf(stderr, "BENCH FAIL: cannot open %s\n", tvpath);
        goto cleanup;
    }
    char magic[4];
    if (!read_exact(f, magic, 4) || memcmp(magic, "TVB1", 4) != 0 ||
        !read_exact(f, &num_samples, 4)) {
        fprintf(stderr, "BENCH FAIL: bad or truncated test-vectors header\n");
        goto cleanup;
    }
    if (num_samples == 0) {
        fprintf(stderr, "BENCH FAIL: test-vectors file has no samples\n");
        goto cleanup;
    }
    samples = calloc(num_samples, sizeof(Sample));
    if (!samples) { fprintf(stderr, "BENCH FAIL: out of memory\n"); goto cleanup; }

    for (uint32_t si = 0; si < num_samples; si++) {
        uint32_t num_inputs = 0;
        if (!read_exact(f, &num_inputs, 4)) {
            fprintf(stderr, "BENCH FAIL: truncated sample %u\n", si);
            goto cleanup;
        }
        Sample *s = &samples[si];
        s->num_inputs = num_inputs;
        s->names = calloc(num_inputs, sizeof(char *));
        s->tensors = calloc(num_inputs, sizeof(OrtValue *));
        s->bufs = calloc(num_inputs, sizeof(void *));
        if (!s->names || !s->tensors || !s->bufs) {
            fprintf(stderr, "BENCH FAIL: out of memory\n");
            goto cleanup;
        }
        for (uint32_t ii = 0; ii < num_inputs; ii++) {
            uint32_t name_len = 0, dcode = 0, ndim = 0;
            if (!read_exact(f, &name_len, 4)) { goto trunc; }
            s->names[ii] = malloc((size_t)name_len + 1);
            if (!s->names[ii] || !read_exact(f, s->names[ii], name_len)) { goto trunc; }
            s->names[ii][name_len] = '\0';
            if (!read_exact(f, &dcode, 4) || !read_exact(f, &ndim, 4)) { goto trunc; }
            int64_t *dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
            if (!dims) { fprintf(stderr, "BENCH FAIL: out of memory\n"); goto cleanup; }
            for (uint32_t d = 0; d < ndim; d++) {
                if (!read_exact(f, &dims[d], 8)) { free(dims); goto trunc; }
            }
            uint64_t nbytes = 0;
            if (!read_exact(f, &nbytes, 8)) { free(dims); goto trunc; }
            s->bufs[ii] = malloc(nbytes ? nbytes : 1);
            if (!s->bufs[ii] || !read_exact(f, s->bufs[ii], nbytes)) { free(dims); goto trunc; }
            OrtStatus *ts = api->CreateTensorWithDataAsOrtValue(
                mem_info, s->bufs[ii], nbytes, dims, ndim, code_to_ort(dcode),
                &s->tensors[ii]);
            free(dims);
            if (ts) {
                fprintf(stderr, "BENCH FAIL: CreateTensor: %s\n", api->GetErrorMessage(ts));
                api->ReleaseStatus(ts);
                goto cleanup;
            }
        }
        /* Reference payload: keep the floats when present so we can report cosine
         * similarity vs the fp32 golden. Empty (ref_count == 0) under --inputs-only. */
        uint64_t ref_count = 0;
        if (!read_exact(f, &ref_count, 8)) { goto trunc; }
        s->ref_count = (size_t)ref_count;
        if (ref_count) {
            s->ref = malloc((size_t)ref_count * sizeof(float));
            if (!s->ref || !read_exact(f, s->ref, (size_t)ref_count * sizeof(float))) {
                goto trunc;
            }
        }
        continue;
    trunc:
        fprintf(stderr, "BENCH FAIL: truncated sample %u\n", si);
        goto cleanup;
    }
    fclose(f);
    f = NULL;

    /* Warmup (untimed), then timed loop, cycling samples. */
    for (long i = 0; i < warmup; i++) {
        Sample *s = &samples[i % num_samples];
        OrtValue *output = NULL;
        ORT_CHECK(api->Run(session, NULL,
                           (const char *const *)s->names,
                           (const OrtValue *const *)s->tensors, s->num_inputs,
                           (const char *const *)&out_name, 1, &output), "Run(warmup)");
        api->ReleaseValue(output);
    }

    times = malloc((size_t)iters * sizeof(double));
    if (!times) { fprintf(stderr, "BENCH FAIL: out of memory\n"); goto cleanup; }
    for (long i = 0; i < iters; i++) {
        Sample *s = &samples[i % num_samples];
        OrtValue *output = NULL;
        double t0 = now_ms();
        ORT_CHECK(api->Run(session, NULL,
                           (const char *const *)s->names,
                           (const OrtValue *const *)s->tensors, s->num_inputs,
                           (const char *const *)&out_name, 1, &output), "Run");
        double t1 = now_ms();
        api->ReleaseValue(output);
        times[i] = t1 - t0;
    }

    /* Correctness (untimed): if the vectors carry an fp32 golden reference, run
     * each sample once more and score its output by cosine similarity. Purely
     * informational — a mismatch or a low score never fails the benchmark. */
    int have_cosine = 0;
    double cosine_sum = 0.0, cosine_min = 1.0;
    long cosine_n = 0;
    for (uint32_t si = 0; si < num_samples; si++) {
        Sample *s = &samples[si];
        if (s->ref_count == 0) continue;
        OrtValue *output = NULL;
        OrtStatus *rs = api->Run(session, NULL,
                                 (const char *const *)s->names,
                                 (const OrtValue *const *)s->tensors, s->num_inputs,
                                 (const char *const *)&out_name, 1, &output);
        if (rs) {
            fprintf(stderr, "BENCH WARN: cosine Run failed: %s\n", api->GetErrorMessage(rs));
            api->ReleaseStatus(rs);
            continue;
        }
        size_t got = 0;
        float *ov = tensor_to_float(api, output, &got);
        if (ov) {
            if (got == s->ref_count) {
                double sim = cosine(ov, s->ref, got);
                have_cosine = 1;
                cosine_sum += sim;
                if (sim < cosine_min) cosine_min = sim;
                cosine_n++;
            } else {
                fprintf(stderr,
                        "BENCH WARN: sample %u output count %zu != ref %zu; skipping cosine\n",
                        si, got, s->ref_count);
            }
            free(ov);
        }
        api->ReleaseValue(output);
    }
    double mean_cosine = cosine_n ? cosine_sum / (double)cosine_n : 0.0;

    double sum = 0.0;
    for (long i = 0; i < iters; i++) sum += times[i];
    double mean_ms = sum / (double)iters;
    qsort(times, (size_t)iters, sizeof(double), cmp_double);
    double p50 = percentile(times, (size_t)iters, 0.50);
    double p90 = percentile(times, (size_t)iters, 0.90);
    double p99 = percentile(times, (size_t)iters, 0.99);
    double throughput_ips = mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0;

    struct rusage ru;
    long peak_rss_kb = 0;
    if (getrusage(RUSAGE_SELF, &ru) == 0) peak_rss_kb = ru.ru_maxrss;

    printf("{\"load_ms\": %.3f, \"iters\": %ld, \"mean_ms\": %.4f, "
           "\"p50_ms\": %.4f, \"p90_ms\": %.4f, \"p99_ms\": %.4f, "
           "\"throughput_ips\": %.2f, \"peak_rss_kb\": %ld",
           load_ms, iters, mean_ms, p50, p90, p99, throughput_ips, peak_rss_kb);
    if (have_cosine) {
        printf(", \"mean_cosine\": %.6f, \"min_cosine\": %.6f", mean_cosine, cosine_min);
    }
    printf("}\n");
    exit_code = 0;

cleanup:
    if (f) fclose(f);
    free(times);
    if (samples) {
        for (uint32_t si = 0; si < num_samples; si++) {
            Sample *s = &samples[si];
            if (s->tensors) {
                for (uint32_t ii = 0; ii < s->num_inputs; ii++)
                    if (s->tensors[ii]) api->ReleaseValue(s->tensors[ii]);
                free(s->tensors);
            }
            if (s->bufs) {
                for (uint32_t ii = 0; ii < s->num_inputs; ii++) free(s->bufs[ii]);
                free(s->bufs);
            }
            if (s->names) {
                for (uint32_t ii = 0; ii < s->num_inputs; ii++) free(s->names[ii]);
                free(s->names);
            }
            free(s->ref);
        }
        free(samples);
    }
    if (out_name) (void)api->AllocatorFree(allocator, out_name);
    if (mem_info) api->ReleaseMemoryInfo(mem_info);
    if (session)  api->ReleaseSession(session);
    if (opts)     api->ReleaseSessionOptions(opts);
    if (env)      api->ReleaseEnv(env);
    return exit_code;
}
