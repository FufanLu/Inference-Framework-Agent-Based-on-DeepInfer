//! Which Kafka offsets are safe to commit when jobs finish out of order.
//!
//! A runner works on many jobs of one partition at once and they finish in
//! any order. Committing the offset of a job that finished while an earlier
//! one is still running would skip the earlier job if the runner died: on
//! restart the group resumes after the committed offset. So the committed
//! position of a partition is its lowest offset still in flight, or one past
//! the highest finished offset when nothing is in flight. Delivery is at
//! least once; the job store's claim turns redeliveries of finished jobs into
//! no-ops.

use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Default)]
struct Partition {
    in_flight: BTreeSet<i64>,
    /// One past the highest offset that finished.
    next_after_done: Option<i64>,
    /// The position committed (or handed out for commit) last.
    committed: Option<i64>,
}

impl Partition {
    fn position(&self) -> Option<i64> {
        match self.in_flight.first() {
            Some(&lowest) => Some(lowest),
            None => self.next_after_done,
        }
    }
}

#[derive(Debug, Default)]
pub struct OffsetTracker {
    partitions: HashMap<i32, Partition>,
}

impl OffsetTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// A message was handed to a runner.
    pub fn start(&mut self, partition: i32, offset: i64) {
        self.partitions
            .entry(partition)
            .or_default()
            .in_flight
            .insert(offset);
    }

    /// A message's outcome is stored. Unknown partitions (revoked meanwhile)
    /// are ignored: their new owner will see the message again.
    pub fn done(&mut self, partition: i32, offset: i64) {
        if let Some(state) = self.partitions.get_mut(&partition)
            && state.in_flight.remove(&offset)
        {
            let next = offset + 1;
            state.next_after_done = Some(state.next_after_done.map_or(next, |n| n.max(next)));
        }
    }

    /// Messages in flight.
    pub fn in_flight(&self) -> usize {
        self.partitions.values().map(|p| p.in_flight.len()).sum()
    }

    /// Positions to commit: `(partition, next offset to consume)` for every
    /// partition whose position advanced since the last call.
    pub fn committable(&mut self) -> Vec<(i32, i64)> {
        let mut out = Vec::new();
        for (&partition, state) in &mut self.partitions {
            if let Some(position) = state.position()
                && state.committed.is_none_or(|committed| position > committed)
            {
                state.committed = Some(position);
                out.push((partition, position));
            }
        }
        out.sort_unstable();
        out
    }

    /// The partition moved to another runner: forget it.
    pub fn revoke(&mut self, partition: i32) -> Option<i64> {
        self.partitions
            .remove(&partition)
            .and_then(|state| state.position())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commits_stop_at_the_lowest_job_in_flight() {
        let mut tracker = OffsetTracker::new();
        for offset in 10..14 {
            tracker.start(0, offset);
        }
        assert_eq!(tracker.committable(), vec![(0, 10)]);
        // 11 and 13 finish first: offset 10 still runs, nothing to commit.
        tracker.done(0, 11);
        tracker.done(0, 13);
        assert_eq!(tracker.committable(), vec![]);
        // 10 finishes: the position moves to the next job still running.
        tracker.done(0, 10);
        assert_eq!(tracker.committable(), vec![(0, 12)]);
        // Everything finished: one past the highest offset.
        tracker.done(0, 12);
        assert_eq!(tracker.committable(), vec![(0, 14)]);
        assert_eq!(tracker.committable(), vec![]);
        assert_eq!(tracker.in_flight(), 0);
    }

    #[test]
    fn partitions_are_independent() {
        let mut tracker = OffsetTracker::new();
        tracker.start(0, 5);
        tracker.start(1, 100);
        tracker.start(1, 101);
        tracker.done(1, 101);
        assert_eq!(tracker.committable(), vec![(0, 5), (1, 100)]);
        tracker.done(1, 100);
        assert_eq!(tracker.committable(), vec![(1, 102)]);
    }

    #[test]
    fn revoked_partitions_are_forgotten() {
        let mut tracker = OffsetTracker::new();
        tracker.start(3, 7);
        tracker.start(3, 8);
        tracker.done(3, 8);
        assert_eq!(tracker.revoke(3), Some(7));
        // A late ack from the old assignment changes nothing.
        tracker.done(3, 7);
        assert_eq!(tracker.committable(), vec![]);
        assert_eq!(tracker.in_flight(), 0);
    }
}
