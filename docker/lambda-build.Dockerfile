# Build environment for compiling ONNX Runtime on AWS Lambda AL2023.
# The build script is injected at run time via a bind mount — nothing is baked in.

FROM public.ecr.aws/lambda/provided:al2023

RUN dnf install -y \
        cmake \
        ninja-build \
        git \
        gcc \
        gcc-c++ \
        ccache \
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
        onnx

ENTRYPOINT ["/bin/bash"]
