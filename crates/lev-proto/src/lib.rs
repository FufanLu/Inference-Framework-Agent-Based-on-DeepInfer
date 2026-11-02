//! Generated gRPC types for `leviathan.worker.v1`.
//!
//! See `proto/leviathan/worker/v1/worker.proto` for the contract, including
//! the block hash convention shared with every worker.

pub mod worker {
    pub mod v1 {
        tonic::include_proto!("leviathan.worker.v1");
    }
}

pub use worker::v1 as pb;
