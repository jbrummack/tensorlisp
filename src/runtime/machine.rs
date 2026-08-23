use lasso::Spur;

//
// Large integer
//

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LargeInt([u8; 7]);

impl From<u64> for LargeInt {
    fn from(value: u64) -> Self {
        let [a, b, c, d, e, f, g, _] = value.to_ne_bytes();
        Self([a, b, c, d, e, f, g])
    }
}

impl From<LargeInt> for u64 {
    fn from(value: LargeInt) -> Self {
        let [a, b, c, d, e, f, g] = value.0;
        Self::from_ne_bytes([a, b, c, d, e, f, g, 0])
    }
}

impl From<i64> for LargeInt {
    fn from(value: i64) -> Self {
        let [a, b, c, d, e, f, g, _] = value.to_ne_bytes();
        Self([a, b, c, d, e, f, g])
    }
}

impl From<LargeInt> for i64 {
    fn from(value: LargeInt) -> Self {
        let [a, b, c, d, e, f, g] = value.0;

        // Sign extend the 56-bit value.
        let sign = if g & 0x80 != 0 { 0xff } else { 0x00 };

        Self::from_ne_bytes([a, b, c, d, e, f, g, sign])
    }
}

//
// Expressions
//

#[derive(Clone, Debug)]
pub enum Expr {
    Nil,

    /// Pointer to another expression slot.
    Medium {
        cell: u32,
        cell_offset: u8,
    },

    Atom(Spur),
    String(u32),
    Bool(bool),
    Float(f32),
    Int(LargeInt),

    Context(LargeInt),
    Tensor(LargeInt),

    Op(Op),
}

impl Expr {
    pub fn medium(cell: usize, offset: usize) -> Self {
        assert!(offset < 4);

        Self::Medium {
            cell: cell as u32,
            cell_offset: offset as u8,
        }
    }
}

//
// Memory
//

#[derive(Debug)]
pub struct FreeMap {
    /// One bit per medium cell.
    ///
    /// 0 = free
    /// 1 = occupied
    pub slots: Vec<u64>,
}

impl FreeMap {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn find_free(&self) -> Option<usize> {
        for (offset, &el) in self.slots.iter().enumerate() {
            if el != u64::MAX {
                let free_bit = (!el).trailing_zeros() as usize;
                return Some((offset * 64) + free_bit);
            }
        }

        None
    }

    pub fn is_used(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;

        self.slots
            .get(word)
            .map(|value| value & (1u64 << bit) != 0)
            .unwrap_or(false)
    }

    pub fn mark_used(&mut self, index: usize) {
        self.ensure(index);

        let word = index / 64;
        let bit = index % 64;

        self.slots[word] |= 1u64 << bit;
    }

    pub fn mark_free(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;

        if let Some(slot) = self.slots.get_mut(word) {
            *slot &= !(1u64 << bit);
        }
    }

    fn ensure(&mut self, index: usize) {
        let word = index / 64;

        while self.slots.len() <= word {
            self.slots.push(0);
        }
    }
}

//
// Runtime
//

pub struct Runtime {
    pub freemap: FreeMap,

    /// Each entry is exactly one cache-line-sized medium.
    pub mediums: Vec<[Expr; 4]>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            freemap: FreeMap::new(),
            mediums: Vec::new(),
        }
    }

    //
    // Medium allocation
    //

    fn alloc_medium(&mut self) -> usize {
        if let Some(cell) = self.freemap.find_free() {
            self.freemap.mark_used(cell);

            // A FreeMap slot can technically refer to memory we haven't
            // allocated in `mediums` yet, so grow the backing store.
            while self.mediums.len() <= cell {
                self.mediums
                    .push([Expr::Nil, Expr::Nil, Expr::Nil, Expr::Nil]);
            }

            self.mediums[cell] = [Expr::Nil, Expr::Nil, Expr::Nil, Expr::Nil];

            return cell;
        }

        let cell = self.mediums.len();

        self.mediums
            .push([Expr::Nil, Expr::Nil, Expr::Nil, Expr::Nil]);

        self.freemap.mark_used(cell);

        cell
    }

    fn free_medium(&mut self, cell: usize) {
        if cell < self.mediums.len() {
            self.mediums[cell] = [Expr::Nil, Expr::Nil, Expr::Nil, Expr::Nil];

            self.freemap.mark_free(cell);
        }
    }

    //
    // Medium access
    //

    fn get_medium(&self, cell: usize) -> &[Expr; 4] {
        &self.mediums[cell]
    }

    fn get_medium_mut(&mut self, cell: usize) -> &mut [Expr; 4] {
        &mut self.mediums[cell]
    }

    fn get_expr<'a>(&'a self, expr: &'a Expr) -> Option<&'a Expr> {
        match expr {
            Expr::Medium { cell, cell_offset } => self
                .mediums
                .get(*cell as usize)
                .and_then(|medium| medium.get(*cell_offset as usize)),

            _ => Some(expr),
        }
    }

    //
    // Lists
    //

    /// Create a one-element list.
    pub fn list1(&mut self, value: Expr) -> Expr {
        let cell = self.alloc_medium();

        self.mediums[cell][0] = value;

        Expr::medium(cell, 0)
    }

    /// Allocate a new list element.
    ///
    /// The first three slots are data.
    /// Slot 3 is reserved for the continuation medium.
    pub fn cons(&mut self, value: Expr, list: Expr) -> Expr {
        let cell = self.alloc_medium();

        self.mediums[cell][0] = value;
        self.mediums[cell][1] = list;

        Expr::medium(cell, 0)
    }

    /// Return the first logical element.
    pub fn car(&self, list: &Expr) -> Option<&Expr> {
        match list {
            Expr::Medium { cell, cell_offset } => {
                let medium = self.mediums.get(*cell as usize)?;
                medium.get(*cell_offset as usize)
            }

            _ => None,
        }
    }

    /// Return the logical remainder.
    pub fn cdr(&self, list: &Expr) -> Option<Expr> {
        match list {
            Expr::Medium { cell, cell_offset } => {
                let medium = self.mediums.get(*cell as usize)?;
                let offset = *cell_offset as usize;

                if offset >= 3 {
                    return None;
                }

                match &medium[offset + 1] {
                    Expr::Nil => None,

                    next => Some(next.clone()),
                }
            }

            _ => None,
        }
    }

    //
    // Chunked lists
    //

    /// Append a value to the end of a chunked medium list.
    ///
    /// Each medium contains:
    ///
    ///     [ value, value, value, next ]
    ///
    /// The fourth slot is therefore the continuation pointer.
    pub fn push(&mut self, list: &Expr, value: Expr) -> Expr {
        let mut current = list.clone();

        loop {
            let (cell, offset) = match current {
                Expr::Medium { cell, cell_offset } => (cell as usize, cell_offset as usize),

                _ => {
                    return self.list1(value);
                }
            };

            // We have three data positions per medium.
            if offset < 2 {
                let next_offset = offset + 1;

                if matches!(self.mediums[cell][next_offset], Expr::Nil) {
                    self.mediums[cell][next_offset] = value;
                    return list.clone();
                }

                current = self.mediums[cell][next_offset].clone();
                continue;
            }

            // offset == 2 means this is the final data slot.
            //
            // Slot 3 is the continuation pointer.
            match self.mediums[cell][3].clone() {
                Expr::Nil => {
                    let next_cell = self.alloc_medium();

                    self.mediums[next_cell][0] = value;
                    self.mediums[cell][3] = Expr::medium(next_cell, 0);

                    return list.clone();
                }

                next => {
                    current = next;
                }
            }
        }
    }

    /// Walk a chunked list and collect its expressions.
    pub fn collect(&self, list: &Expr) -> Vec<Expr> {
        let mut result = Vec::new();

        let mut current = match list {
            Expr::Medium { .. } => list.clone(),
            _ => return result,
        };

        loop {
            let (cell, offset) = match current {
                Expr::Medium { cell, cell_offset } => (cell as usize, cell_offset as usize),

                _ => break,
            };

            let Some(medium) = self.mediums.get(cell) else {
                break;
            };

            if offset >= 4 {
                break;
            }

            // Consume data slots until the continuation slot.
            for index in offset..3 {
                match &medium[index] {
                    Expr::Nil => break,

                    value => result.push(value.clone()),
                }
            }

            // Fourth slot is the continuation.
            match &medium[3] {
                Expr::Medium { .. } => {
                    current = medium[3].clone();
                }

                _ => break,
            }
        }

        result
    }

    //
    // Debugging
    //

    pub fn dump(&self, list: &Expr) {
        let values = self.collect(list);

        println!("(");

        for value in values {
            println!("    {:?}", value);
        }

        println!(")");
    }
}

//
// Operations
//

#[derive(Clone, Copy, Debug)]
pub enum Op {
    IsExpr,
    IsList,

    Car,
    Cdr,
    Cons,

    Quote,
    Eq,
    Cond,

    Lambda,
    If,
    Def,

    Math(Math),
}

#[derive(Clone, Copy, Debug)]
pub enum Math {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}
