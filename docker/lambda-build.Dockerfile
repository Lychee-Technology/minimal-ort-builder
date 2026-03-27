# Build environment for compiling ONNX Runtime targeting AWS Lambda AL2023.
# Uses the standard AL2023 base image (not the Lambda provided image) to avoid
# Lambda runtime init behaviour that can cause docker run to hang.
# The build script is injected at run time via a bind mount — nothing is baked in.

FROM public.ecr.aws/amazonlinux/amazonlinux:2023

RUN dnf install -y --allowerasing \
        cmake \
        ninja-build \
        git \
        gcc \
        gcc-c++ \
        jq \
        curl \
        xz \
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
        onnx

# ccache is not in AL2023 repos and the PyPI package has no arm64 wheel.
# Install the pre-built arm64 binary from the official ccache GitHub release.
ARG CCACHE_VERSION=4.10.2
RUN curl -fsSL \
      "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-aarch64.tar.xz" \
      -o /tmp/ccache.tar.xz \
    && tar -xf /tmp/ccache.tar.xz -C /usr/local/bin \
         --strip-components=1 \
         "ccache-${CCACHE_VERSION}-linux-aarch64/ccache" \
    && rm /tmp/ccache.tar.xz

ENTRYPOINT ["/bin/bash"]
