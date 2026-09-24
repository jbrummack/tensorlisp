pub enum MediumList<T> {
    Empty,
    One([T; 1]),
    Two([T; 2]),
    Three([T; 3]),
    Four([T; 4]),
    Five([T; 5]),
}

pub struct FreeMap {
    slots: Vec<u64>,
    max_slot: usize,
}

impl FreeMap {
    pub fn new(size: usize) -> Self {
        Self {
            // Allocate initial storage for 512 u64 blocks (32,768 total bits/slots)
            slots: vec![0; size / 64],
            max_slot: 0,
        }
    }

    /// Marks a bit as allocated (1). Automatically expands vector if bit out of bounds.
    fn set_full(&mut self, bit_idx: usize) {
        let slot = bit_idx / 64;
        let bit = bit_idx % 64;

        if slot >= self.slots.len() {
            self.slots.resize(slot + 1, 0);
        }

        self.slots[slot] |= 1 << bit;
    }

    /// Marks a bit as free (0).
    fn set_free(&mut self, bit_idx: usize) {
        let slot = bit_idx / 64;
        let bit = bit_idx % 64;

        if slot < self.slots.len() {
            self.slots[slot] &= !(1 << bit);
        }
    }

    /// Finds the first available bit, marks it full, and returns its index.
    pub fn allocate(&mut self) -> usize {
        let r = if let Some(bit_idx) = self.find_free() {
            self.set_full(bit_idx);
            bit_idx
        } else {
            // If all current slots are full, append a new word and allocate the first bit
            let bit_idx = self.slots.len() * 64;
            self.slots.push(1); // Set bit 0 of the new u64 word
            bit_idx
        };
        self.max_slot = self.max_slot.max(r);
        r
    }
    pub fn max(&self) -> usize {
        self.max_slot
    }
    /// Finds the index of the first zero bit (free slot).
    pub fn find_free(&self) -> Option<usize> {
        for (offset, &el) in self.slots.iter().enumerate() {
            if el != u64::MAX {
                let free_bit = (!el).trailing_zeros() as usize;
                return Some((offset * 64) + free_bit);
            }
        }
        None
    }
}
pub struct Allocator<T> {
    pub slots: FreeMap,
    pub entries: Vec<T>,
}
