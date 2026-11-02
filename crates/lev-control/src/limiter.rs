//! Token-bucket rate limiting, in-process or shared through Redis.
//!
//! A bucket refills at `per_minute / 60` units per second up to `per_minute`.
//! Requests per minute use [`Limiter::acquire`] (take one unit or wait). Tokens
//! per minute cannot be known before a request finishes, so admission only
//! requires a positive balance ([`Limiter::available`]) and the actual usage is
//! charged afterwards with [`Limiter::debit`], which may drive the balance
//! negative and so delay the tenant's next requests.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;

#[async_trait]
pub trait Limiter: Send + Sync + std::fmt::Debug {
    /// Take `amount` units, or return how long until they would be available.
    async fn acquire(&self, key: &str, amount: f64, per_minute: u64) -> Result<(), Duration>;
    /// Current balance (may be negative).
    async fn available(&self, key: &str, per_minute: u64) -> f64;
    /// Charge `amount` units unconditionally.
    async fn debit(&self, key: &str, amount: f64, per_minute: u64);
}

fn rate_per_s(per_minute: u64) -> f64 {
    per_minute as f64 / 60.0
}

/// Time until a bucket at `balance` holds `amount`.
pub fn wait_for(balance: f64, amount: f64, per_minute: u64) -> Duration {
    let missing = (amount - balance).max(0.0);
    Duration::from_secs_f64(missing / rate_per_s(per_minute).max(f64::MIN_POSITIVE))
}

#[derive(Debug, Clone, Copy)]
struct Bucket {
    tokens: f64,
    updated: Instant,
}

/// Buckets in this process: exact for a single gateway.
#[derive(Debug, Default)]
pub struct LocalLimiter {
    buckets: Mutex<HashMap<String, Bucket>>,
}

impl LocalLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    fn with_bucket<R>(&self, key: &str, per_minute: u64, f: impl FnOnce(&mut Bucket) -> R) -> R {
        let capacity = per_minute as f64;
        let now = Instant::now();
        let mut buckets = self.buckets.lock();
        let bucket = buckets.entry(key.to_string()).or_insert(Bucket {
            tokens: capacity,
            updated: now,
        });
        let elapsed = now.duration_since(bucket.updated).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate_per_s(per_minute)).min(capacity);
        bucket.updated = now;
        f(bucket)
    }
}

#[async_trait]
impl Limiter for LocalLimiter {
    async fn acquire(&self, key: &str, amount: f64, per_minute: u64) -> Result<(), Duration> {
        self.with_bucket(key, per_minute, |bucket| {
            if bucket.tokens >= amount {
                bucket.tokens -= amount;
                Ok(())
            } else {
                Err(wait_for(bucket.tokens, amount, per_minute))
            }
        })
    }

    async fn available(&self, key: &str, per_minute: u64) -> f64 {
        self.with_bucket(key, per_minute, |bucket| bucket.tokens)
    }

    async fn debit(&self, key: &str, amount: f64, per_minute: u64) {
        self.with_bucket(key, per_minute, |bucket| bucket.tokens -= amount);
    }
}

#[cfg(feature = "redis")]
pub use redis_limiter::RedisLimiter;

#[cfg(feature = "redis")]
mod redis_limiter {
    use super::*;

    /// One atomic bucket operation. Uses the Redis server clock, so gateways
    /// with skewed clocks agree. mode: 0 acquire, 1 debit, 2 peek.
    const SCRIPT: &str = r#"
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local amount = tonumber(ARGV[3])
local mode = tonumber(ARGV[4])
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1]) or capacity
local ts = tonumber(data[2]) or now
tokens = math.min(capacity, tokens + (now - ts) * rate)
local ok = 1
if mode == 0 then
  if tokens >= amount then tokens = tokens - amount else ok = 0 end
elseif mode == 1 then
  tokens = tokens - amount
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
return {ok, tostring(tokens)}
"#;

    /// Buckets in Redis, shared by every gateway replica.
    #[derive(Clone)]
    pub struct RedisLimiter {
        connection: redis::aio::ConnectionManager,
        script: redis::Script,
        prefix: String,
    }

    impl std::fmt::Debug for RedisLimiter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RedisLimiter")
                .field("prefix", &self.prefix)
                .finish_non_exhaustive()
        }
    }

    impl RedisLimiter {
        pub async fn connect(url: &str) -> anyhow::Result<Self> {
            let client = redis::Client::open(url)?;
            let connection = redis::aio::ConnectionManager::new(client).await?;
            Ok(Self {
                connection,
                script: redis::Script::new(SCRIPT),
                prefix: "lev:bucket:".to_string(),
            })
        }

        async fn call(&self, key: &str, amount: f64, per_minute: u64, mode: u8) -> (bool, f64) {
            let mut connection = self.connection.clone();
            let result: redis::RedisResult<(i64, String)> = self
                .script
                .key(format!("{}{key}", self.prefix))
                .arg(per_minute as f64 / 60_000.0)
                .arg(per_minute as f64)
                .arg(amount)
                .arg(mode)
                .invoke_async(&mut connection)
                .await;
            match result {
                Ok((ok, tokens)) => (ok == 1, tokens.parse().unwrap_or(0.0)),
                Err(error) => {
                    // Fail open: an unavailable limiter must not take the API down.
                    tracing::warn!(%error, "redis rate limiter unavailable; allowing request");
                    (true, per_minute as f64)
                }
            }
        }
    }

    #[async_trait]
    impl Limiter for RedisLimiter {
        async fn acquire(&self, key: &str, amount: f64, per_minute: u64) -> Result<(), Duration> {
            match self.call(key, amount, per_minute, 0).await {
                (true, _) => Ok(()),
                (false, tokens) => Err(wait_for(tokens, amount, per_minute)),
            }
        }

        async fn available(&self, key: &str, per_minute: u64) -> f64 {
            self.call(key, 0.0, per_minute, 2).await.1
        }

        async fn debit(&self, key: &str, amount: f64, per_minute: u64) {
            self.call(key, amount, per_minute, 1).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bucket_allows_burst_then_waits() {
        let limiter = LocalLimiter::new();
        for _ in 0..60 {
            assert!(limiter.acquire("t", 1.0, 60).await.is_ok());
        }
        let wait = limiter.acquire("t", 1.0, 60).await.unwrap_err();
        assert!(wait > Duration::from_millis(900) && wait <= Duration::from_secs(1));
    }

    #[tokio::test]
    async fn debit_can_go_negative() {
        let limiter = LocalLimiter::new();
        limiter.debit("t", 150.0, 100).await;
        assert!(limiter.available("t", 100).await < 0.0);
        let wait = limiter.acquire("t", 1.0, 100).await.unwrap_err();
        // 51 units missing at 100/60 per second: about 30.6 s.
        assert!(wait > Duration::from_secs(30) && wait < Duration::from_secs(31));
    }
}
