//! Size-bucketed pools of device buffers: session state and step arenas are
//! taken from here and handed back on drop, so a finished session's memory
//! serves the next one that fits.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::backend::{Backend, MemKind};
use crate::Result;

/// Rounds up to one of four sizes per power of two (a 25% worst-case waste),
/// at least 4 KiB.
pub fn size_class(bytes: u64) -> u64 {
    const MIN: u64 = 4096;
    let b = bytes.max(MIN);
    let top = b.next_power_of_two();
    let step = top / 8; // quarters of the octave below `top`
    let mut c = top / 2 + step;
    while c < b {
        c += step;
    }
    c.min(top)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Buffers created.
    pub allocated: u64,
    /// Requests served from the free lists.
    pub reused: u64,
    /// Bytes in buffers currently parked in the free lists.
    pub idle_bytes: u64,
    /// Bytes ever allocated.
    pub total_bytes: u64,
}

struct State<B: Backend> {
    free: BTreeMap<u64, Vec<B::Buffer>>,
    stats: PoolStats,
}

pub struct BlockPool<B: Backend> {
    backend: Arc<B>,
    kind: MemKind,
    /// A free block up to this many times the class of the request may serve it.
    slack: u64,
    zero: bool,
    state: Arc<Mutex<State<B>>>,
}

/// A buffer on loan from a pool; goes back on drop.
pub struct Block<B: Backend> {
    buffer: Option<B::Buffer>,
    class: u64,
    home: Arc<Mutex<State<B>>>,
}

impl<B: Backend> Block<B> {
    pub fn buffer(&self) -> &B::Buffer {
        self.buffer.as_ref().expect("a block holds its buffer until dropped")
    }

    /// Size class (the buffer is at least this big).
    pub fn class(&self) -> u64 {
        self.class
    }
}

impl<B: Backend> Drop for Block<B> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut st = self.home.lock().unwrap();
            st.stats.idle_bytes += self.class;
            st.free.entry(self.class).or_default().push(buffer);
        }
    }
}

impl<B: Backend> BlockPool<B> {
    /// `zero`: clear every block when it is handed out (session state must
    /// start at zero; step arenas are fully written before they are read).
    pub fn new(backend: Arc<B>, kind: MemKind, slack: u64, zero: bool) -> Self {
        BlockPool {
            backend,
            kind,
            slack: slack.max(1),
            zero,
            state: Arc::new(Mutex::new(State { free: BTreeMap::new(), stats: PoolStats::default() })),
        }
    }

    pub fn stats(&self) -> PoolStats {
        self.state.lock().unwrap().stats
    }

    pub fn take(&self, bytes: u64) -> Result<Block<B>> {
        let class = size_class(bytes.max(1));
        let reused = {
            let mut st = self.state.lock().unwrap();
            let hit = st.free.range(class..=class.saturating_mul(self.slack)).find(|(_, v)| !v.is_empty()).map(|(c, _)| *c);
            hit.map(|c| {
                let buffer = st.free.get_mut(&c).unwrap().pop().unwrap();
                st.stats.reused += 1;
                st.stats.idle_bytes -= c;
                (buffer, c)
            })
        };
        let (buffer, class) = match reused {
            Some(r) => r,
            None => {
                let buffer = self.backend.alloc(class, self.kind)?;
                let mut st = self.state.lock().unwrap();
                st.stats.allocated += 1;
                st.stats.total_bytes += class;
                (buffer, class)
            }
        };
        if self.zero {
            // A fresh Vulkan allocation is not zeroed either.
            self.backend.zero(&buffer, 0, self.backend.buffer_len(&buffer) & !3)?;
        }
        Ok(Block { buffer: Some(buffer), class, home: self.state.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_round_up_with_bounded_waste() {
        assert_eq!(size_class(1), 4096);
        assert_eq!(size_class(4096), 4096);
        assert_eq!(size_class(4097), 5120);
        assert_eq!(size_class(1 << 20), 1 << 20);
        for n in [5000u64, 70_000, 1_000_000, 123_456_789, (3 << 30) + 1] {
            let c = size_class(n);
            assert!(c >= n && c as f64 <= n as f64 * 1.26, "{n} -> {c}");
        }
        // Monotone.
        let mut prev = 0;
        for n in (1..200_000).step_by(997) {
            let c = size_class(n);
            assert!(c >= prev);
            prev = c;
        }
    }
}
