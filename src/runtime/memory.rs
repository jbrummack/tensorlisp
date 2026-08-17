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
}
impl FreeMap {
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
