# syntax=docker/dockerfile:1.7
FROM rust:1.97-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY proto proto
COPY crates crates
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p lev-gateway -p lev-bench \
    && cp target/release/lev-gateway target/release/lev-bench /usr/local/bin/

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/lev-gateway /usr/local/bin/lev-bench /usr/local/bin/
COPY assets /opt/leviathan/assets
WORKDIR /opt/leviathan
EXPOSE 8080
ENTRYPOINT ["lev-gateway"]
