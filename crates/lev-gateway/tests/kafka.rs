//! The Kafka job queue against a real broker. Set `LEV_TEST_KAFKA_BROKERS`
//! (e.g. `127.0.0.1:9092`) to run; skipped otherwise.
#![cfg(feature = "kafka")]

use std::time::Duration;

use lev_gateway::config::KafkaConfig;
use lev_gateway::jobs::kafka::KafkaQueue;
use lev_gateway::jobs::{Endpoint, JobMessage, JobQueue};
use uuid::Uuid;

fn brokers() -> Option<String> {
    std::env::var("LEV_TEST_KAFKA_BROKERS").ok()
}

fn config(brokers: &str, name: &str) -> KafkaConfig {
    KafkaConfig {
        brokers: brokers.to_string(),
        topic: format!("lev.test.{name}"),
        group: format!("lev-test-{name}"),
        partitions: 1,
        ..KafkaConfig::default()
    }
}

fn job(n: usize) -> JobMessage {
    JobMessage {
        id: Uuid::now_v7(),
        tenant_id: "acme".to_string(),
        endpoint: Endpoint::ChatCompletions,
        body: serde_json::json!({ "n": n }),
        submitted_at_ms: 0,
    }
}

async fn next(queue: &KafkaQueue) -> lev_gateway::jobs::queue::Delivery {
    tokio::time::timeout(Duration::from_secs(30), queue.next())
        .await
        .expect("a job within 30 s")
        .unwrap()
}

/// A runner dies with jobs unacknowledged: the next runner of the group
/// resumes at the lowest unfinished job, not after the highest finished one.
#[tokio::test]
async fn unacknowledged_jobs_are_redelivered_to_the_next_runner() {
    let Some(brokers) = brokers() else {
        eprintln!("LEV_TEST_KAFKA_BROKERS not set; skipping");
        return;
    };
    let name = Uuid::new_v4().simple().to_string();
    let config = config(&brokers, &name);
    let jobs: Vec<JobMessage> = (0..5).map(job).collect();

    let first = KafkaQueue::connect(&config, true).await.unwrap();
    first.enqueue(&jobs).await.unwrap();
    let d0 = next(&first).await;
    let d1 = next(&first).await;
    let d2 = next(&first).await;
    assert_eq!(
        [&d0.job, &d1.job, &d2.job].map(|job| job.id),
        [jobs[0].id, jobs[1].id, jobs[2].id]
    );
    // Jobs 0 and 2 finish, job 1 is still running when the runner dies.
    d0.ack();
    d2.ack();
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let stats = first.stats().await;
    assert_eq!(stats["partitions"][0]["committed"], 1, "{stats}");
    drop(d1);
    drop(first);

    let second = KafkaQueue::connect(&config, true).await.unwrap();
    let redelivered: Vec<Uuid> = {
        let mut ids = Vec::new();
        for _ in 0..4 {
            let delivery = next(&second).await;
            ids.push(delivery.job.id);
            delivery.ack();
        }
        ids
    };
    assert_eq!(
        redelivered,
        vec![jobs[1].id, jobs[2].id, jobs[3].id, jobs[4].id],
        "resumes at job 1; job 2 is delivered again and the job store turns it into a no-op"
    );
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    let stats = second.stats().await;
    assert_eq!(stats["backlog"], 0, "{stats}");
}

/// One tenant's burst spreads over every partition, so every runner of the
/// group shares it.
#[tokio::test]
async fn one_tenants_burst_spreads_over_partitions() {
    let Some(brokers) = brokers() else {
        eprintln!("LEV_TEST_KAFKA_BROKERS not set; skipping");
        return;
    };
    let name = Uuid::new_v4().simple().to_string();
    let config = KafkaConfig {
        partitions: 6,
        ..config(&brokers, &name)
    };
    let queue = KafkaQueue::connect(&config, true).await.unwrap();
    let jobs: Vec<JobMessage> = (0..120).map(job).collect();
    queue.enqueue(&jobs).await.unwrap();
    let stats = queue.stats().await;
    let partitions = stats["partitions"].as_array().unwrap();
    assert_eq!(partitions.len(), 6, "{stats}");
    assert!(
        partitions
            .iter()
            .all(|p| p["high_watermark"].as_i64().unwrap() > 0),
        "every partition got some of the burst: {stats}"
    );
    assert_eq!(stats["backlog"], 120, "{stats}");
    let mut seen = std::collections::HashSet::new();
    for _ in 0..jobs.len() {
        let delivery = next(&queue).await;
        seen.insert(delivery.job.id);
        delivery.ack();
    }
    assert_eq!(seen.len(), jobs.len());
}
