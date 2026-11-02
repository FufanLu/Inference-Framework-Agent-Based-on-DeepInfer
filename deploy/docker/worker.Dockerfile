# syntax=docker/dockerfile:1.7
# The sim worker image. The vLLM worker runs on GPU nodes from a conda
# environment instead (deploy/slurm).
FROM python:3.12-slim
WORKDIR /opt/leviathan
RUN pip install --no-cache-dir "grpcio>=1.70" "grpcio-tools>=1.70" "protobuf>=5.29" \
    "tokenizers>=0.21" "xxhash>=3.5"
COPY proto proto
COPY assets assets
COPY worker worker
RUN mkdir -p worker/leviathan_worker/_gen && python -m grpc_tools.protoc -I proto \
    --python_out=worker/leviathan_worker/_gen --grpc_python_out=worker/leviathan_worker/_gen \
    --pyi_out=worker/leviathan_worker/_gen proto/leviathan/worker/v1/worker.proto
ENV PYTHONPATH=/opt/leviathan/worker PYTHONUNBUFFERED=1
EXPOSE 50051
ENTRYPOINT ["python", "-m", "leviathan_worker.server"]
