//! Generational handles: what the control plane (Scheme) holds instead of
//! pointers. A freed slot bumps its generation, so a stale handle is an error
//! rather than a use-after-free, and the table owns the value, so a GC of the
//! handle never frees device memory by itself.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Handle {
    pub index: u32,
    pub generation: u32,
}

impl Handle {
    /// Packs into one integer (fits a Scheme fixnum on 64-bit targets).
    pub fn to_u64(self) -> u64 {
        (self.generation as u64) << 32 | self.index as u64
    }

    pub fn from_u64(v: u64) -> Handle {
        Handle { index: v as u32, generation: (v >> 32) as u32 }
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

pub struct HandleTable<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> Default for HandleTable<T> {
    fn default() -> Self {
        HandleTable { slots: Vec::new(), free: Vec::new() }
    }
}

impl<T> HandleTable<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, value: T) -> Handle {
        match self.free.pop() {
            Some(index) => {
                let slot = &mut self.slots[index as usize];
                slot.value = Some(value);
                Handle { index, generation: slot.generation }
            }
            None => {
                self.slots.push(Slot { generation: 0, value: Some(value) });
                Handle { index: self.slots.len() as u32 - 1, generation: 0 }
            }
        }
    }

    pub fn get(&self, h: Handle) -> Option<&T> {
        let slot = self.slots.get(h.index as usize)?;
        if slot.generation == h.generation { slot.value.as_ref() } else { None }
    }

    pub fn get_mut(&mut self, h: Handle) -> Option<&mut T> {
        let slot = self.slots.get_mut(h.index as usize)?;
        if slot.generation == h.generation { slot.value.as_mut() } else { None }
    }

    /// Removes the value; the handle (and every copy of it) becomes invalid.
    pub fn remove(&mut self, h: Handle) -> Option<T> {
        let slot = self.slots.get_mut(h.index as usize)?;
        if slot.generation != h.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(h.index);
        Some(value)
    }

    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_handles_fail_cleanly() {
        let mut t = HandleTable::new();
        let a = t.insert("a");
        assert_eq!(t.get(a), Some(&"a"));
        assert_eq!(t.remove(a), Some("a"));
        assert_eq!(t.get(a), None);
        assert_eq!(t.remove(a), None);
        // The slot is reused, with a new generation.
        let b = t.insert("b");
        assert_eq!(b.index, a.index);
        assert_ne!(b.generation, a.generation);
        assert_eq!(t.get(a), None);
        assert_eq!(t.get(b), Some(&"b"));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn packing_round_trips() {
        let h = Handle { index: 0xdead, generation: 0xbeef };
        assert_eq!(Handle::from_u64(h.to_u64()), h);
    }
}
