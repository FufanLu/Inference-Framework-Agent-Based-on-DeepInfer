//! Integration tests against real Postgres and Redis.
//!
//! Skipped unless the services are given:
//! `LEV_TEST_DATABASE_URL=postgres://lev:lev@127.0.0.1:55432/lev LEV_TEST_REDIS_URL=redis://127.0.0.1:56379 cargo test -p lev-control --test services`

use std::time::Duration;

use chrono::Utc;
use lev_control::limiter::RedisLimiter;
use lev_control::pg::PgStore;
use lev_control::{Limiter, Limits, Store, Tenant, UsageRecord, keys};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

#[tokio::test]
async fn postgres_store_round_trip() {
    let Some(url) = env("LEV_TEST_DATABASE_URL") else {
        eprintln!("LEV_TEST_DATABASE_URL not set; skipping");
        return;
    };
    let store = PgStore::connect(&url).await.unwrap();
    let id = format!("t-{}", uuid::Uuid::new_v4().simple());
    let tenant = Tenant {
        id: id.clone(),
        name: "Integration".into(),
        limits: Limits {
            rpm: Some(60),
            tpm: None,
            monthly_token_quota: Some(1_000_000),
        },
        disabled: false,
    };
    store.upsert_tenant(&tenant).await.unwrap();
    assert_eq!(store.get_tenant(&id).await.unwrap(), Some(tenant.clone()));

    let key = keys::generate();
    let info = store
        .insert_key(&id, "ci", &keys::display_prefix(&key), &keys::hash(&key))
        .await
        .unwrap();
    // Idempotent on the same hash.
    let again = store
        .insert_key(&id, "ci", &keys::display_prefix(&key), &keys::hash(&key))
        .await
        .unwrap();
    assert_eq!(info.id, again.id);
    let (found, found_tenant) = store.lookup_key(&keys::hash(&key)).await.unwrap().unwrap();
    assert_eq!(found.id, info.id);
    assert_eq!(found_tenant, tenant);
    assert_eq!(store.list_keys(Some(&id)).await.unwrap().len(), 1);

    let record = |prompt: u64, completion: u64| UsageRecord {
        tenant_id: id.clone(),
        request_id: uuid::Uuid::new_v4().to_string(),
        model: "DeepSeek-V4.1-Flash".into(),
        prompt_tokens: prompt,
        cached_tokens: prompt / 2,
        completion_tokens: completion,
        at: Utc::now(),
    };
    store
        .record_usage(&[record(100, 10), record(200, 20)])
        .await
        .unwrap();
    let totals = store
        .usage_since(&id, Utc::now() - chrono::Duration::hours(1))
        .await
        .unwrap();
    assert_eq!(
        (
            totals.requests,
            totals.prompt_tokens,
            totals.cached_tokens,
            totals.completion_tokens
        ),
        (2, 300, 150, 30)
    );
    let daily = store
        .daily_usage(Some(&id), Utc::now() - chrono::Duration::days(1))
        .await
        .unwrap();
    assert_eq!(daily.len(), 1);
    assert_eq!(daily[0].totals.requests, 2);

    assert!(store.revoke_key(info.id).await.unwrap());
    assert!(store.lookup_key(&keys::hash(&key)).await.unwrap().is_none());
}

#[tokio::test]
async fn redis_limiter_is_shared_and_refills() {
    let Some(url) = env("LEV_TEST_REDIS_URL") else {
        eprintln!("LEV_TEST_REDIS_URL not set; skipping");
        return;
    };
    // Two limiters stand in for two gateway replicas.
    let a = RedisLimiter::connect(&url).await.unwrap();
    let b = RedisLimiter::connect(&url).await.unwrap();
    let key = format!("test-{}", uuid::Uuid::new_v4().simple());
    // 120 per minute = 2 per second, burst 120.
    for i in 0..120 {
        let limiter = if i % 2 == 0 { &a } else { &b };
        assert!(limiter.acquire(&key, 1.0, 120).await.is_ok(), "request {i}");
    }
    let wait = a.acquire(&key, 1.0, 120).await.unwrap_err();
    assert!(wait <= Duration::from_millis(600), "{wait:?}");
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(b.acquire(&key, 1.0, 120).await.is_ok());
    b.debit(&key, 500.0, 120).await;
    assert!(a.available(&key, 120).await < 0.0);
}
