//! Which prompt prefixes each worker holds in its KV cache.
//!
//! Every block hash (see [`crate::hashing`]) identifies a whole prefix, so the
//! index maps hashes to the set of worker slots that cache that block, split
//! by tier. A request's cached prefix on a worker is the longest run of its
//! block hashes, starting at the first block, that the worker holds.
//!
//! Two sources feed the index:
//! - **exact**: KV events published by the worker (`BlockStored` /
//!   `BlockRemoved`), the ground truth of its prefix cache;
//! - **approximate**: the router's own record of the prompts it sent to a
//!   worker, kept in a per-worker shadow LRU with a TTL and a capacity. It
//!   serves backends that publish no events, and it is the baseline against
//!   which exact routing is benchmarked.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// A set of worker slots. Slot `i` is bit `i`.
pub type SlotMask = u128;

/// The maximum number of workers one router tracks.
pub const MAX_SLOTS: usize = 128;

/// KV cache tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Medium {
    Gpu,
    Cpu,
}

#[derive(Debug, Default, Clone, Copy)]
struct Entry {
    gpu: SlotMask,
    cpu: SlotMask,
}

impl Entry {
    fn any(self) -> SlotMask {
        self.gpu | self.cpu
    }
}

/// The cached prefix of one request on one worker, in blocks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Matched blocks resident in GPU memory.
    pub gpu_blocks: usize,
    /// Matched blocks that would be loaded from the CPU tier.
    pub cpu_blocks: usize,
}

impl PrefixMatch {
    pub fn blocks(self) -> usize {
        self.gpu_blocks + self.cpu_blocks
    }
}

/// Per-worker shadow LRU for approximate mode.
#[derive(Debug)]
struct Shadow {
    capacity: usize,
    ttl: Duration,
    /// hash -> generation of its latest insertion.
    live: HashMap<u64, (u64, Instant)>,
    /// Insertion order; stale generations are skipped lazily.
    order: VecDeque<(u64, u64)>,
    generation: u64,
}

impl Shadow {
    fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity: capacity.max(1),
            ttl,
            live: HashMap::new(),
            order: VecDeque::new(),
            generation: 0,
        }
    }

    /// Record `hashes` as cached now. Returns hashes newly added.
    fn touch(&mut self, hashes: &[u64], now: Instant) -> Vec<u64> {
        let mut added = Vec::new();
        for &hash in hashes {
            self.generation += 1;
            let previous = self.live.insert(hash, (self.generation, now));
            if previous.is_none() {
                added.push(hash);
            }
            self.order.push_back((hash, self.generation));
        }
        added
    }

    /// Drop expired entries and entries beyond capacity. Returns the dropped
    /// hashes.
    fn expire(&mut self, now: Instant) -> Vec<u64> {
        let mut dropped = Vec::new();
        while let Some(&(hash, generation)) = self.order.front() {
            let Some(&(live_generation, inserted)) = self.live.get(&hash) else {
                self.order.pop_front();
                continue;
            };
            if live_generation != generation {
                // A newer insertion of the same hash is further back.
                self.order.pop_front();
                continue;
            }
            let over_capacity = self.live.len() > self.capacity;
            let expired = now.duration_since(inserted) >= self.ttl;
            if !over_capacity && !expired {
                break;
            }
            self.order.pop_front();
            self.live.remove(&hash);
            dropped.push(hash);
        }
        // Bound the order queue even when the same hashes are touched
        // repeatedly.
        if self.order.len() > self.live.len() * 4 + 1024 {
            let live = &self.live;
            self.order
                .retain(|(hash, generation)| live.get(hash).is_some_and(|(g, _)| g == generation));
        }
        dropped
    }

    fn len(&self) -> usize {
        self.live.len()
    }
}

/// Configuration of the approximate (shadow) index.
#[derive(Debug, Clone, Copy)]
pub struct ShadowConfig {
    /// Blocks remembered per worker; defaults to the worker's KV capacity.
    pub capacity_blocks: Option<usize>,
    pub ttl: Duration,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            capacity_blocks: None,
            ttl: Duration::from_secs(300),
        }
    }
}

/// The prefix index over all worker slots.
#[derive(Debug, Default)]
pub struct PrefixIndex {
    entries: HashMap<u64, Entry>,
    /// Blocks per slot and tier, for statistics.
    counts: HashMap<(usize, Medium), usize>,
    shadows: HashMap<usize, Shadow>,
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn bit(slot: usize) -> SlotMask {
        debug_assert!(slot < MAX_SLOTS);
        1u128 << slot
    }

    fn mask_mut(entry: &mut Entry, medium: Medium) -> &mut SlotMask {
        match medium {
            Medium::Gpu => &mut entry.gpu,
            Medium::Cpu => &mut entry.cpu,
        }
    }

    /// Record that `slot` stores `hashes` in `medium` (exact mode).
    pub fn store(&mut self, slot: usize, hashes: &[u64], medium: Medium) {
        let bit = Self::bit(slot);
        for &hash in hashes {
            let entry = self.entries.entry(hash).or_default();
            let mask = Self::mask_mut(entry, medium);
            if *mask & bit == 0 {
                *mask |= bit;
                *self.counts.entry((slot, medium)).or_default() += 1;
            }
        }
    }

    /// Record that `slot` no longer stores `hashes` in `medium` (exact mode).
    pub fn remove(&mut self, slot: usize, hashes: &[u64], medium: Medium) {
        let bit = Self::bit(slot);
        for hash in hashes {
            let Some(entry) = self.entries.get_mut(hash) else {
                continue;
            };
            let mask = Self::mask_mut(entry, medium);
            if *mask & bit != 0 {
                *mask &= !bit;
                if let Some(count) = self.counts.get_mut(&(slot, medium)) {
                    *count = count.saturating_sub(1);
                }
            }
            if entry.any() == 0 {
                self.entries.remove(hash);
            }
        }
    }

    /// Forget everything `slot` caches, in both tiers and the shadow index.
    pub fn clear_slot(&mut self, slot: usize) {
        let bit = Self::bit(slot);
        self.entries.retain(|_, entry| {
            entry.gpu &= !bit;
            entry.cpu &= !bit;
            entry.any() != 0
        });
        self.counts.retain(|(s, _), _| *s != slot);
        self.shadows.remove(&slot);
    }

    /// Approximate mode: record that `hashes` were just sent to `slot`.
    pub fn shadow_touch(
        &mut self,
        slot: usize,
        hashes: &[u64],
        capacity: usize,
        config: ShadowConfig,
        now: Instant,
    ) {
        let capacity = config.capacity_blocks.unwrap_or(capacity);
        let shadow = self
            .shadows
            .entry(slot)
            .or_insert_with(|| Shadow::new(capacity, config.ttl));
        let added = shadow.touch(hashes, now);
        let dropped = shadow.expire(now);
        self.store(slot, &added, Medium::Gpu);
        self.remove(slot, &dropped, Medium::Gpu);
    }

    /// Approximate mode: expire stale shadow entries of every slot.
    pub fn shadow_expire(&mut self, now: Instant) {
        let mut dropped_by_slot = Vec::new();
        for (&slot, shadow) in &mut self.shadows {
            let dropped = shadow.expire(now);
            if !dropped.is_empty() {
                dropped_by_slot.push((slot, dropped));
            }
        }
        for (slot, dropped) in dropped_by_slot {
            self.remove(slot, &dropped, Medium::Gpu);
        }
    }

    /// Longest cached prefix of `hashes` on every slot in `candidates`.
    ///
    /// Returns one entry per candidate slot, in ascending slot order.
    pub fn match_prefix(&self, hashes: &[u64], candidates: SlotMask) -> Vec<(usize, PrefixMatch)> {
        let mut result: HashMap<usize, PrefixMatch> = slots(candidates)
            .map(|slot| (slot, PrefixMatch::default()))
            .collect();
        let mut alive = candidates;
        for hash in hashes {
            if alive == 0 {
                break;
            }
            let entry = self.entries.get(hash).copied().unwrap_or_default();
            alive &= entry.any();
            for slot in slots(alive) {
                let matched = result.get_mut(&slot).expect("slot is a candidate");
                if entry.gpu & Self::bit(slot) != 0 {
                    matched.gpu_blocks += 1;
                } else {
                    matched.cpu_blocks += 1;
                }
            }
        }
        let mut result: Vec<_> = result.into_iter().collect();
        result.sort_unstable_by_key(|(slot, _)| *slot);
        result
    }

    /// Blocks recorded for `slot` in `medium`.
    pub fn blocks(&self, slot: usize, medium: Medium) -> usize {
        self.counts.get(&(slot, medium)).copied().unwrap_or(0)
    }

    /// Entries in the shadow index of `slot`.
    pub fn shadow_blocks(&self, slot: usize) -> usize {
        self.shadows.get(&slot).map_or(0, Shadow::len)
    }

    /// Distinct block hashes across all workers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Iterate the slots set in `mask`, ascending.
pub fn slots(mask: SlotMask) -> impl Iterator<Item = usize> {
    let mut remaining = mask;
    std::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }
        let slot = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        Some(slot)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(n: usize) -> SlotMask {
        (0..n).fold(0, |mask, slot| mask | (1u128 << slot))
    }

    #[test]
    fn longest_prefix_stops_at_first_gap() {
        let mut index = PrefixIndex::new();
        index.store(0, &[1, 2, 3], Medium::Gpu);
        index.store(1, &[1, 3], Medium::Gpu);
        let matches = index.match_prefix(&[1, 2, 3, 4], all(2));
        assert_eq!(matches[0].1.blocks(), 3);
        // Slot 1 lacks block 2, so block 3 does not count.
        assert_eq!(matches[1].1.blocks(), 1);
    }

    #[test]
    fn tiers_are_tracked_separately() {
        let mut index = PrefixIndex::new();
        index.store(0, &[1], Medium::Gpu);
        index.store(0, &[2], Medium::Cpu);
        let matched = index.match_prefix(&[1, 2], all(1))[0].1;
        assert_eq!(
            matched,
            PrefixMatch {
                gpu_blocks: 1,
                cpu_blocks: 1
            }
        );
        index.remove(0, &[2], Medium::Cpu);
        assert_eq!(index.match_prefix(&[1, 2], all(1))[0].1.blocks(), 1);
        assert_eq!(index.blocks(0, Medium::Cpu), 0);
    }

    #[test]
    fn clear_slot_forgets_only_that_slot() {
        let mut index = PrefixIndex::new();
        index.store(0, &[1, 2], Medium::Gpu);
        index.store(1, &[1, 2], Medium::Gpu);
        index.clear_slot(0);
        let matches = index.match_prefix(&[1, 2], all(2));
        assert_eq!(matches[0].1.blocks(), 0);
        assert_eq!(matches[1].1.blocks(), 2);
        assert_eq!(index.blocks(0, Medium::Gpu), 0);
    }

    #[test]
    fn duplicate_store_counts_once() {
        let mut index = PrefixIndex::new();
        index.store(0, &[7, 7, 7], Medium::Gpu);
        assert_eq!(index.blocks(0, Medium::Gpu), 1);
        index.remove(0, &[7], Medium::Gpu);
        assert!(index.is_empty());
    }

    #[test]
    fn shadow_respects_capacity_and_ttl() {
        let mut index = PrefixIndex::new();
        let config = ShadowConfig {
            capacity_blocks: Some(2),
            ttl: Duration::from_secs(10),
        };
        let t0 = Instant::now();
        index.shadow_touch(0, &[1, 2, 3], 100, config, t0);
        // Capacity 2: the oldest block is gone.
        assert_eq!(index.shadow_blocks(0), 2);
        assert_eq!(index.match_prefix(&[1, 2, 3], all(1))[0].1.blocks(), 0);
        assert_eq!(index.match_prefix(&[2, 3], all(1))[0].1.blocks(), 2);
        index.shadow_expire(t0 + Duration::from_secs(11));
        assert_eq!(index.shadow_blocks(0), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn shadow_touch_refreshes_existing_entries() {
        let mut index = PrefixIndex::new();
        let config = ShadowConfig {
            capacity_blocks: Some(2),
            ttl: Duration::from_secs(10),
        };
        let t0 = Instant::now();
        index.shadow_touch(0, &[1, 2], 100, config, t0);
        index.shadow_touch(0, &[1], 100, config, t0 + Duration::from_secs(1));
        index.shadow_touch(0, &[3], 100, config, t0 + Duration::from_secs(2));
        // 2 is the least recently touched.
        assert_eq!(index.match_prefix(&[1], all(1))[0].1.blocks(), 1);
        assert_eq!(index.match_prefix(&[2], all(1))[0].1.blocks(), 0);
        assert_eq!(index.match_prefix(&[3], all(1))[0].1.blocks(), 1);
    }

    #[test]
    fn slots_iterates_set_bits() {
        assert_eq!(slots(0b1011).collect::<Vec<_>>(), vec![0, 1, 3]);
        assert_eq!(slots(1u128 << 127).collect::<Vec<_>>(), vec![127]);
    }
}
