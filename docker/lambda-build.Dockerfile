# Build environment for compiling ONNX Runtime targeting AWS Lambda AL2023.
# Uses the standard AL2023 base image (not the Lambda provided image) to avoid
# Lambda runtime init behaviour that can cause docker run to hang.
# The build script is injected at run time via a bind mount — nothing is baked in.

FROM public.ecr.aws/amazonlinux/amazonlinux:2023

RUN dnf install -y \
        cmake \
        ninja-build \
        git \
        gcc \
        gcc-c++ \
        jq \
        python3 \
        python3-pip \
        tar \
        gzip \
        which \
    && dnf clean all

RUN pip3 install --no-cache-dir \
        "huggingface_hub[cli]" \
        numpy \
        sympy \
        packaging \
        onnx \
        ccache

ENTRYPOINT ["/bin/bash"]
