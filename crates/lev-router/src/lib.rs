//! Leviathan request router.
//!
//! DeepInfer's open-source release leaves load balancing across data-parallel
//! replicas and prefill/decode disaggregation to the serving layer. This crate
//! is that layer's decision core: it tracks which prompt prefixes each worker
//! holds in its KV cache and picks workers for every request.

pub mod hashing;
