//! The block hash convention shared by the router and every worker.
//!
//! Only full blocks of `block_size` prompt tokens are hashed. Each block hash
//! seeds the next one, so a hash identifies the whole prefix up to and
//! including its block:
//!
//! ```text
//! h[0] = xxh3_64_with_seed(le_u32_bytes(tokens[0..B]),        ROOT_SEED)
//! h[i] = xxh3_64_with_seed(le_u32_bytes(tokens[iB..(i+1)B]),  h[i-1])
//! ```
//!
//! The Python worker implements the same function
//! (`leviathan_worker.hashing.block_hashes`); both test suites check the same
//! vectors.

use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Seed of the first block of every prompt: the bytes of `"LEVIATHA"`.
pub const ROOT_SEED: u64 = 0x4C45_5649_4154_4841;

/// Return the chain hash of every full block of `tokens`.
pub fn block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    assert!(block_size > 0, "block_size must be positive");
    let mut hashes = Vec::with_capacity(tokens.len() / block_size);
    let mut parent = ROOT_SEED;
    let mut bytes = Vec::with_capacity(block_size * 4);
    for block in tokens.chunks_exact(block_size) {
        bytes.clear();
        for token in block {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        parent = xxh3_64_with_seed(&bytes, parent);
        hashes.push(parent);
    }
    hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors shared with `worker/tests/test_hashing.py`.
    pub(crate) const VECTOR_BLOCK_SIZE: usize = 4;

    fn vector_tokens() -> Vec<u32> {
        (0..10).collect()
    }

    #[test]
    fn hashes_only_full_blocks() {
        assert_eq!(block_hashes(&vector_tokens(), VECTOR_BLOCK_SIZE).len(), 2);
        assert!(block_hashes(&[1, 2, 3], VECTOR_BLOCK_SIZE).is_empty());
    }

    #[test]
    fn hash_identifies_the_whole_prefix() {
        let a = block_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        let b = block_hashes(&[9, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_ne!(a[0], b[0]);
        // Same second block, different first block: the chain differs.
        assert_ne!(a[1], b[1]);
        let c = block_hashes(&[1, 2, 3, 4, 0, 0, 0, 0], 4);
        assert_eq!(a[0], c[0]);
        assert_ne!(a[1], c[1]);
    }

    #[test]
    fn matches_cross_language_vector() {
        let hashes = block_hashes(&vector_tokens(), VECTOR_BLOCK_SIZE);
        assert_eq!(hashes, CROSS_LANGUAGE_VECTOR);
    }

    /// Expected hashes of tokens `0..10` with block size 4; the Python test
    /// asserts the same values.
    const CROSS_LANGUAGE_VECTOR: [u64; 2] = [0x05CA_F1E6_F5C0_6121, 0x075F_F973_CE99_7C6B];
}
