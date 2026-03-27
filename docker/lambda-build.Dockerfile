# Build environment for compiling ONNX Runtime targeting AWS Lambda AL2023.
# Uses the standard AL2023 base image (not the Lambda provided image) to avoid
# Lambda runtime init behaviour that can cause docker run to hang.
# Uses Clang 20 + lld 20 (available in AL2023 repos) instead of GCC 11 to avoid
# the -mcpu=neoverse-n1 / -march=armv8-a conflict present in GCC 11.
# The build script is injected at run time via a bind mount — nothing is baked in.

FROM public.ecr.aws/amazonlinux/amazonlinux:2023

RUN dnf install -y --allowerasing \
        ninja-build \
        git \
        clang20 \
        lld20 \
        jq \
        curl \
        xz \
        patch \
        python3.11 \
        python3.11-pip \
        tar \
        gzip \
        which \
    && dnf clean all \
    && ln -sf /usr/bin/clang-20    /usr/local/bin/clang \
    && ln -sf /usr/bin/clang++-20  /usr/local/bin/clang++ \
    && ln -sf /usr/bin/ld.lld-20   /usr/local/bin/ld.lld \
    && ln -sf /usr/bin/python3.11  /usr/local/bin/python3 \
    && ln -sf /usr/bin/pip3.11     /usr/local/bin/pip3

# Ensure /usr/local/bin (python3/pip3 symlinks + pip-installed scripts) comes first.
ENV PATH="/usr/local/bin:$PATH"

RUN pip3 install --no-cache-dir \
        "cmake>=3.28,<4" \
        "huggingface_hub[cli]>=0.21,<1.0" \
        numpy \
        sympy \
        packaging \
        onnx \
        flatbuffers \
        "onnxruntime==1.24.4"

# ccache is not in AL2023 repos and the PyPI package has no arm64 wheel.
# Use the musl-static build so it runs on any glibc version without linking issues.
ARG CCACHE_VERSION=4.13.2
RUN curl -fsSL \
      "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-aarch64-musl-static.tar.gz" \
      -o /tmp/ccache.tar.gz \
    && tar -xzf /tmp/ccache.tar.gz -C /tmp \
    && mv /tmp/ccache-${CCACHE_VERSION}-linux-aarch64-musl-static/ccache /usr/local/bin/ccache \
    && chmod +x /usr/local/bin/ccache \
    && rm -rf /tmp/ccache.tar.gz /tmp/ccache-${CCACHE_VERSION}-linux-aarch64-musl-static

# Sanity-check: all required tools must be on PATH before we ship the image.
RUN clang --version && clang++ --version \
    && which huggingface-cli && huggingface-cli --help \
    && which hf && hf --help \
    && which ccache && ccache --version

ENTRYPOINT ["/bin/bash"]
