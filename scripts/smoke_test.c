/*
 * smoke_test.c — minimal ORT session loader.
 * Usage: ./smoke_test <path/to/model.onnx>
 * Exit 0 = model loaded successfully, non-zero = failure.
 */
#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime_c_api.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.onnx>\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];

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
    OrtStatus *status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "smoke_test", &env);
    if (status) {
        fprintf(stderr, "SMOKE FAIL: CreateEnv: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return 1;
    }

    OrtSessionOptions *opts = NULL;
    status = api->CreateSessionOptions(&opts);
    if (status) {
        fprintf(stderr, "SMOKE FAIL: CreateSessionOptions: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        api->ReleaseEnv(env);
        return 1;
    }

    OrtSession *session = NULL;
    status = api->CreateSession(env, model_path, opts, &session);
    if (status) {
        fprintf(stderr, "SMOKE FAIL: CreateSession(%s): %s\n", model_path, api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        api->ReleaseSessionOptions(opts);
        api->ReleaseEnv(env);
        return 1;
    }

    fprintf(stdout, "SMOKE OK: loaded %s\n", model_path);
    fflush(stdout);

    api->ReleaseSession(session);
    api->ReleaseSessionOptions(opts);
    api->ReleaseEnv(env);
    return 0;
}
