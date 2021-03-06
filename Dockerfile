# This docker base is built based on Dockerfile.kungfu-base in the same folder.
FROM swr.cn-north-1.myhuaweicloud.com/mailuo/kungfu-docker:v.12 as base

# Make sure you clone the source code locally before build.
ADD . /KungFu

# Install KungFu
RUN cd KungFu && \
        ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
        pip install --no-index --user -U . && \
        ldconfig && \
        GOBIN=$(pwd)/bin go install -v ./...

WORKDIR /