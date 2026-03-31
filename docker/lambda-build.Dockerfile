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

RUN pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.7.0+cpu"

# Sanity-check: all required tools must be on PATH before we ship the image.
RUN clang --version && clang++ --version \
    && which huggingface-cli && huggingface-cli --help \
    && which hf && hf --help \
    && python3 -c "import torch; import onnxruntime; from onnxruntime.transformers import optimizer; print(torch.__version__)"

ENTRYPOINT ["/bin/bash"]
