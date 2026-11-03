//! Kafka job queue.
//!
//! - **Producer**: idempotent, `acks=all`, keyed by job ID so that jobs
//!   spread evenly over partitions and therefore over runners. (Keying by
//!   tenant would keep a tenant's jobs in order, which independent jobs do not
//!   need, and would put one tenant's whole burst on one partition: one
//!   runner.) A submission returns only after Kafka acknowledged every job.
//! - **Consumer**: one consumer group of job runners (`jobs.kafka.group`);
//!   each partition belongs to one runner at a time, so partitions bound how
//!   many gateways share the work. Offsets are committed by hand, only up to
//!   the lowest job still running ([`OffsetTracker`]): at-least-once delivery
//!   without skipping a job when a runner dies mid-batch.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use parking_lot::Mutex;
use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
use rdkafka::client::DefaultClientContext;
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{
    BaseConsumer, CommitMode, Consumer, ConsumerContext, Rebalance, StreamConsumer,
};
use rdkafka::error::RDKafkaErrorCode;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::{ClientContext, Message, Offset, TopicPartitionList};
use serde_json::json;

use super::offsets::OffsetTracker;
use super::queue::{Delivery, JobMessage, JobQueue};
use crate::config::KafkaConfig;

const COMMIT_INTERVAL: Duration = Duration::from_millis(500);
const SEND_TIMEOUT: Duration = Duration::from_secs(10);

/// Commits what is safe before a partition moves to another runner.
struct RunnerContext {
    topic: String,
    tracker: Arc<Mutex<OffsetTracker>>,
}

impl ClientContext for RunnerContext {}

impl ConsumerContext for RunnerContext {
    fn pre_rebalance(&self, consumer: &BaseConsumer<Self>, rebalance: &Rebalance<'_>) {
        let Rebalance::Revoke(revoked) = rebalance else {
            return;
        };
        let mut positions = TopicPartitionList::new();
        {
            let mut tracker = self.tracker.lock();
            for element in revoked.elements() {
                if let Some(position) = tracker.revoke(element.partition()) {
                    let _ = positions.add_partition_offset(
                        &self.topic,
                        element.partition(),
                        Offset::Offset(position),
                    );
                }
            }
        }
        if positions.count() > 0
            && let Err(error) = consumer.commit(&positions, CommitMode::Sync)
        {
            tracing::warn!(%error, "committing revoked partitions failed");
        }
        tracing::info!(partitions = revoked.count(), "job partitions revoked");
    }

    fn post_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance<'_>) {
        if let Rebalance::Assign(assigned) = rebalance {
            tracing::info!(partitions = assigned.count(), "job partitions assigned");
        }
    }
}

pub struct KafkaQueue {
    topic: String,
    group: String,
    producer: FutureProducer,
    consumer: Option<Arc<StreamConsumer<RunnerContext>>>,
    /// A separate client for lag queries: blocking metadata calls on the
    /// consumer that the runner polls take seconds.
    stats_client: Arc<BaseConsumer>,
    tracker: Arc<Mutex<OffsetTracker>>,
}

impl std::fmt::Debug for KafkaQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KafkaQueue")
            .field("topic", &self.topic)
            .field("group", &self.group)
            .field("consumer", &self.consumer.is_some())
            .finish()
    }
}

fn client_config(config: &KafkaConfig) -> ClientConfig {
    let mut client = ClientConfig::new();
    client
        .set("bootstrap.servers", &config.brokers)
        .set("client.id", &config.client_id);
    for (key, value) in &config.properties {
        client.set(key, value);
    }
    client
}

impl KafkaQueue {
    /// Connect, create the topic if it is missing and, when `consume` is
    /// set, join the runners' consumer group.
    pub async fn connect(config: &KafkaConfig, consume: bool) -> anyhow::Result<Self> {
        ensure_topic(config).await?;
        let producer: FutureProducer = client_config(config)
            .set("enable.idempotence", "true")
            .set("acks", "all")
            .set("linger.ms", "5")
            .set("compression.type", "lz4")
            .set("message.timeout.ms", "30000")
            .create()
            .context("creating the Kafka producer")?;
        let tracker = Arc::new(Mutex::new(OffsetTracker::new()));
        let consumer = if consume {
            let consumer: StreamConsumer<RunnerContext> = client_config(config)
                .set("group.id", &config.group)
                .set("enable.auto.commit", "false")
                .set("auto.offset.reset", "earliest")
                .set("enable.partition.eof", "false")
                .set("session.timeout.ms", "10000")
                // The runner stops reading while the cluster has no room for
                // jobs; that pause must not cost it its partitions.
                .set("max.poll.interval.ms", "900000")
                .create_with_context(RunnerContext {
                    topic: config.topic.clone(),
                    tracker: Arc::clone(&tracker),
                })
                .context("creating the Kafka consumer")?;
            consumer
                .subscribe(&[&config.topic])
                .context("subscribing to the job topic")?;
            let consumer = Arc::new(consumer);
            tokio::spawn(commit_loop(
                Arc::downgrade(&consumer),
                Arc::clone(&tracker),
                config.topic.clone(),
            ));
            Some(consumer)
        } else {
            None
        };
        let stats_client: BaseConsumer = client_config(config)
            .set("group.id", &config.group)
            .set("enable.auto.commit", "false")
            .create()
            .context("creating the Kafka stats client")?;
        tracing::info!(
            brokers = %config.brokers,
            topic = %config.topic,
            group = %config.group,
            consume,
            "Kafka job queue ready"
        );
        Ok(Self {
            topic: config.topic.clone(),
            group: config.group.clone(),
            producer,
            consumer,
            stats_client: Arc::new(stats_client),
            tracker,
        })
    }
}

async fn ensure_topic(config: &KafkaConfig) -> anyhow::Result<()> {
    let admin: AdminClient<DefaultClientContext> = client_config(config)
        .create()
        .context("creating the Kafka admin client")?;
    let topic = NewTopic::new(
        &config.topic,
        config.partitions.max(1),
        TopicReplication::Fixed(config.replication_factor.max(1)),
    );
    let options = AdminOptions::new().operation_timeout(Some(Duration::from_secs(10)));
    let results = admin
        .create_topics([&topic], &options)
        .await
        .context("creating the job topic")?;
    for result in results {
        match result {
            Ok(_) | Err((_, RDKafkaErrorCode::TopicAlreadyExists)) => {}
            Err((name, code)) => anyhow::bail!("creating topic `{name}` failed: {code}"),
        }
    }
    Ok(())
}

async fn commit_loop(
    consumer: std::sync::Weak<StreamConsumer<RunnerContext>>,
    tracker: Arc<Mutex<OffsetTracker>>,
    topic: String,
) {
    let mut interval = tokio::time::interval(COMMIT_INTERVAL);
    loop {
        interval.tick().await;
        let Some(consumer) = consumer.upgrade() else {
            break;
        };
        let positions = tracker.lock().committable();
        if positions.is_empty() {
            continue;
        }
        let mut list = TopicPartitionList::new();
        for (partition, offset) in positions {
            let _ = list.add_partition_offset(&topic, partition, Offset::Offset(offset));
        }
        if let Err(error) = consumer.commit(&list, CommitMode::Async) {
            tracing::warn!(%error, "committing job offsets failed");
        }
    }
}

#[async_trait]
impl JobQueue for KafkaQueue {
    fn backend(&self) -> &'static str {
        "kafka"
    }

    async fn enqueue(&self, jobs: &[JobMessage]) -> anyhow::Result<()> {
        let payloads: Vec<(String, Vec<u8>)> = jobs
            .iter()
            .map(|job| Ok((job.id.to_string(), serde_json::to_vec(job)?)))
            .collect::<anyhow::Result<_>>()?;
        let sends = payloads.iter().map(|(key, payload)| {
            self.producer.send(
                FutureRecord::to(&self.topic).key(key).payload(payload),
                SEND_TIMEOUT,
            )
        });
        for result in futures::future::join_all(sends).await {
            if let Err((error, _)) = result {
                anyhow::bail!("Kafka did not accept a job: {error}");
            }
        }
        Ok(())
    }

    async fn next(&self) -> anyhow::Result<Delivery> {
        let consumer = self
            .consumer
            .as_ref()
            .context("this gateway does not run jobs (jobs.run = false)")?;
        loop {
            let message = consumer.recv().await?;
            let (partition, offset) = (message.partition(), message.offset());
            let Some(job) = message
                .payload()
                .and_then(|payload| serde_json::from_slice::<JobMessage>(payload).ok())
            else {
                tracing::error!(partition, offset, "skipping an unreadable job message");
                let mut tracker = self.tracker.lock();
                tracker.start(partition, offset);
                tracker.done(partition, offset);
                continue;
            };
            self.tracker.lock().start(partition, offset);
            let tracker = Arc::clone(&self.tracker);
            return Ok(Delivery::new(
                job,
                Some(Box::new(move || tracker.lock().done(partition, offset))),
            ));
        }
    }

    async fn stats(&self) -> serde_json::Value {
        let in_flight = self.tracker.lock().in_flight();
        let assigned: Vec<i32> = self
            .consumer
            .as_ref()
            .and_then(|consumer| consumer.assignment().ok())
            .map(|list| list.elements().iter().map(|e| e.partition()).collect())
            .unwrap_or_default();
        let client = Arc::clone(&self.stats_client);
        let topic = self.topic.clone();
        let partitions = tokio::task::spawn_blocking(move || group_lag(&client, &topic))
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default();
        let lag: i64 = partitions.iter().filter_map(|p| p["lag"].as_i64()).sum();
        json!({
            "backend": "kafka",
            "topic": self.topic,
            "group": self.group,
            "consumer": self.consumer.is_some(),
            "assigned_partitions": assigned,
            "in_flight": in_flight,
            "backlog": lag,
            "partitions": partitions,
        })
    }
}

/// The group's lag on every partition of the topic: high watermark minus
/// committed offset (all jobs not yet finished, queued or running).
fn group_lag(client: &BaseConsumer, topic: &str) -> anyhow::Result<Vec<serde_json::Value>> {
    let timeout = Duration::from_secs(2);
    let metadata = client.fetch_metadata(Some(topic), timeout)?;
    let mut list = TopicPartitionList::new();
    for partition in metadata
        .topics()
        .iter()
        .filter(|t| t.name() == topic)
        .flat_map(|t| t.partitions())
    {
        list.add_partition(topic, partition.id());
    }
    let committed = client.committed_offsets(list.clone(), timeout)?;
    let mut out = Vec::new();
    for element in list.elements_for_topic(topic) {
        let partition = element.partition();
        let (low, high) = client.fetch_watermarks(topic, partition, timeout)?;
        let committed = committed
            .find_partition(topic, partition)
            .and_then(|e| e.offset().to_raw())
            .filter(|offset| *offset >= 0);
        out.push(json!({
            "partition": partition,
            "high_watermark": high,
            "committed": committed,
            "lag": (high - committed.unwrap_or(low)).max(0),
        }));
    }
    Ok(out)
}
