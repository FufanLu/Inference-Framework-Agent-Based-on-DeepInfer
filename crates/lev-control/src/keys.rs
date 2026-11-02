//! API key generation and hashing.
//!
//! Keys are 32 random alphanumeric characters (about 190 bits) behind a
//! `lev-` prefix. With that much entropy a single SHA-256 is a safe way to
//! store them: slow password hashes such as argon2 exist to protect
//! low-entropy human passwords from guessing, and would cost tens of
//! milliseconds on every request here.

use rand::Rng;
use rand::distr::Alphanumeric;
use sha2::{Digest, Sha256};

pub const KEY_PREFIX: &str = "lev-";
const SECRET_LEN: usize = 32;
/// Characters of the secret shown in listings.
const DISPLAY_LEN: usize = 6;

/// SHA-256 of a key: what the store keeps.
pub type KeyHash = [u8; 32];

pub fn generate() -> String {
    let secret: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(SECRET_LEN)
        .map(char::from)
        .collect();
    format!("{KEY_PREFIX}{secret}")
}

pub fn hash(key: &str) -> KeyHash {
    Sha256::digest(key.as_bytes()).into()
}

/// The displayable start of a key, e.g. `lev-3fA9xQ…`.
pub fn display_prefix(key: &str) -> String {
    let shown: String = key.chars().take(KEY_PREFIX.len() + DISPLAY_LEN).collect();
    format!("{shown}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_prefixed_and_hash_stably() {
        let a = generate();
        let b = generate();
        assert_ne!(a, b);
        assert!(a.starts_with(KEY_PREFIX));
        assert_eq!(a.len(), KEY_PREFIX.len() + SECRET_LEN);
        assert_eq!(hash(&a), hash(&a));
        assert_ne!(hash(&a), hash(&b));
        assert_eq!(display_prefix("lev-abcdefghij"), "lev-abcdef…");
    }
}
