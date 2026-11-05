# Linux test image for the engine, for machines without Triton (macOS):
# runs the Triton kernels under TRITON_INTERPRET=1 on CPU.
#   docker build -f engine/docker/test.Dockerfile -t lev-engine-test engine
#   docker run --rm -v "$PWD":/src -w /src lev-engine-test make test-engine PYTHON=/usr/local/bin/python
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends make gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir triton "transformers>=4.57" safetensors xxhash numpy pytest
ENV TRITON_INTERPRET=1
