use std::collections::HashSet;

use lasso::{Rodeo, Spur};

use ggml_sys::ffi::ggml_tensor;

use crate::{
    stupid_lisp::Interior::Nil,
    input::ListInput,
    parser::{List, Parser},
    runtime::memory::FreeMap,
};
#[derive(Debug, Hash, PartialEq, Clone, Copy)]
pub enum Leaf {
    Nil,
    End,
    Int(i32),
    Float([u8; 4]),
    Bool(bool),
    Ptr(u32),
    Lambda(Builtin),
    Keyword(Spur),
    String(Spur),
}
impl Leaf {
    pub fn typename(&self) -> &'static str {
        match self {
            Leaf::Nil => "nil",
            Leaf::End => "end",
            Leaf::Int(_) => "int",
            Leaf::Float(_) => "float",
            Leaf::Bool(_) => "bool",
            Leaf::Ptr(_) => "ptr",
            Leaf::Lambda(builtin) => "builtin fn",
            Leaf::Keyword(spur) => "keyword fn",
            Leaf::String(spur) => "string",
        }
    }
}
#[derive(Debug, Hash, PartialEq, Clone, Copy)]
pub enum Builtin {
    Let,
    If,
    Not,
    True,  //(x,y)->x
    False, //(x,y)->y
    Add,   //(x,y)->z
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Fn,
}
#[derive(Debug, Clone, Copy)]
pub struct Res(i32);
pub enum Ptr {
    Leaf(u32),
    List(u32),
}
impl Res {
    pub fn leaf(ptr: usize) -> Self {
        let ptr = ptr as i32;
        Self(ptr)
    }
    pub fn list(ptr: usize) -> Self {
        let ptr = ptr as i32;
        Self(ptr * -1)
    }
    pub fn ptr(self) -> Ptr {
        let s = self.0;
        if s.is_positive() {
            Ptr::Leaf(s as u32)
        } else {
            Ptr::List((s * -1) as u32)
        }
    }
}
#[derive(Debug)]
pub enum Input {
    List(Vec<Self>),
    Leaf(Leaf),
}
#[derive(Debug, Clone, Copy)]
pub enum Val {
    Leaf(Leaf),
    List(u8, [Res; 8]),
}
#[derive(Debug, Clone, Copy)]
pub struct Construct {
    car: Val,
    next: Val,
}
pub struct Lisp {
    leafs: Vec<Leaf>,
    leaflist: FreeMap,
    lists: Vec<[Res; 8]>,
    listlist: FreeMap,
    strings: Rodeo,
    keywords: Rodeo,
}
impl Lisp {
    pub fn allocate_seed_list(&mut self, arr: &[Input]) -> Res {
        let mut buffer = [Res(0); 8];
        for i in 0..arr.len() {
            buffer[i] = self.allocate_input(&arr[i]);
        }
        let seed = self.alloc_list(buffer);
        println!("SEED: {seed:?}");
        seed
    }
    pub fn allocate_list_with_remainder(&mut self, data: [Res; 7], seed: Res) -> Res {
        todo!()
    }
    pub fn allocate_input(&mut self, input: &Input) -> Res {
        match input {
            Input::List(inputs) => self.allocate_list(inputs),
            Input::Leaf(leaf) => self.alloc_leaf(*leaf),
        }
    }
    pub fn allocate_list(&mut self, arr: &[Input]) -> Res {
        let (seed, chunx) = arr.as_rchunks::<7>();
        let mut seed = self.allocate_seed_list(seed);
        /*for x in seed {
            println!("seed {x:?}");
        }*/

        for x in chunx {
            let mut chunk = [seed; 8];
            for (i, x) in x.iter().enumerate() {
                chunk[i] = self.allocate_input(x);
            }
            seed = self.alloc_list(chunk);
            println!("cnx {x:?}");
        }
        todo!()
    }
    pub fn allocate_array(&mut self, arr: &[Input]) -> Res {
        for input in arr {
            let r = match input {
                Input::List(inputs) => self.allocate_list(inputs),
                Input::Leaf(leaf) => self.alloc_leaf(*leaf),
            };
            println!("{r:?}");
        }
        todo!()
        /*for i in arr {
            match i {
                Input::List(inputs) => todo!(),
                Input::Leaf(leaf) => self.alloc_leaf(*leaf),
            }
        }*/
    }
    pub fn new(mut inp: ListInput) -> (Self, Vec<Res>) {
        let inputs = inp.parse_all().unwrap();

        let mut s = Self {
            leafs: vec![Leaf::End; 8096],
            leaflist: FreeMap::new(8096),
            lists: vec![[Res(0); 8]; 8096],
            listlist: FreeMap::new(8096),
            strings: inp.strings,
            keywords: inp.interner,
        };
        let mut starts = Vec::new();
        println!("{inputs:?}");
        s.allocate_array(&inputs);
        //s.load(&Input::Leaf(Leaf::Nil));
        /*for input in inputs {
                    let start = s.load(&input);
                    starts.push(start);
                }
        */
        (s, starts)
    }

    pub fn alloc_leaf(&mut self, leaf: Leaf) -> Res {
        let ptr = self.leaflist.allocate();
        self.leafs[ptr] = leaf;
        Res::leaf(ptr)
    }
    pub fn alloc_list(&mut self, list: [Res; 8]) -> Res {
        let ptr = self.listlist.allocate();
        //println!("alloc_list: ({list:?}) -> ({ptr})");
        self.lists[ptr] = list;
        println!("{ptr}");
        Res::list(ptr)
    }
    fn alloc_carry_over(&mut self, co: &[Res]) -> Res {
        let mut list = [Res(0); 8];
        for (i, res) in co.iter().enumerate() {
            list[i] = *res;
        }
        //print!("carryover {co:?} ");
        let ptr = self.alloc_list(list);
        //println!("-> {ptr:?}");

        ptr
    }

    pub fn write_list(&mut self, list: &[Input]) -> Res {
        /*let ptr = (addr * -1) as usize;
        let carry_over = self.lists[ptr][7];*/
        let (chunks, rest) = list.as_chunks::<7>();
        let mut restbuf = [Res(0); 8];
        let restlen = rest.len();
        for r in 0..restlen {
            restbuf[r] = self.load(&rest[r]);
        }
        let mut seed = self.alloc_carry_over(&restbuf[0..restlen]);
        println!("first: {seed:?} {restbuf:?}");
        for chunk in chunks {
            let mut chunkbuf = [Res(0); 8];
            for (idx, input) in chunk.iter().enumerate() {
                chunkbuf[idx] = self.load(input);
            }
            chunkbuf[7] = seed;
            println!("chunkbuf {chunkbuf:?}");
            seed = self.alloc_list(chunkbuf);

            /*let [a, b, c, d, e, f, g] = chunk.map(|i| self.load(&i));
            seed = self.alloc_list([*a, *b, *c, *d, *e, *f, *g, seed]);*/
        }
        println!("seed: {seed:?}");
        seed
    }
    pub fn load(&mut self, list: &Input) -> Res {
        match list {
            Input::List(inputs) => self.write_list(inputs),
            Input::Leaf(leaf) => self.alloc_leaf(*leaf),
        }
    }
    pub fn fetch(&self, entry_point: Res) -> Val {
        let val = match entry_point.ptr() {
            Ptr::Leaf(lep) => Val::Leaf(self.leafs[lep as usize]),
            Ptr::List(lip) => Val::List(0, self.lists[lip as usize]),
        };
        println!("fetching {entry_point:?} -> {val:?}");
        val
    }
    pub fn next(&self, val: Val) -> Option<Val> {
        match val {
            Val::Leaf(_leaf) => None,
            Val::List(n, r) => Some(Val::List(n + 1, r)),
        }
    }
    pub fn add(&mut self, val: Val) -> Option<Val> {
        let v1 = self.integer(val.clone());
        let val2 = self.next(val)?;
        let v2 = self.integer(val2);
        Some(Val::Leaf(Leaf::Int(v1 + v2)))
    }
    pub fn integer(&self, val: Val) -> i32 {
        match val {
            Val::Leaf(Leaf::Int(i)) => i,
            Val::List(idx, list) => {
                let res = list[idx as usize];
                let int = self.fetch(res);
                self.integer(int)
            }
            _ => panic!("type error expected int"),
        }
    }
    pub fn eval(&mut self, entry_point: Res) -> Computation {
        let current = self.fetch(entry_point);
        let mut scope = Scope {
            runtime: self,
            current: Some(current),
        };
        for _ in 0..16 {
            if let Some(next) = scope.next() {
                println!("{next:?}")
            }
            //println!("{next:?}");
        }

        todo!()
    }
}

pub struct Scope<'a> {
    runtime: &'a mut Lisp,
    current: Option<Val>,
}
impl<'a> Scope<'a> {
    //pub fn pust(&mut self)
    pub fn next(&mut self) -> Option<Val> {
        let current = self.current?;
        match current {
            Val::Leaf(leaf) => {
                self.current = None;
                Some(Val::Leaf(leaf))
            }
            Val::List(7, list) => {
                let ix = 7usize;
                let carry = list[7];

                if carry.0 == 0 {
                    self.current = None;
                    return None;
                }

                // Follow the continuation list and start at index 0.
                let next = self.runtime.fetch(carry);

                match next {
                    Val::List(_, next_list) => {
                        self.current = Some(Val::List(0, next_list));
                        self.next()
                    }

                    // This shouldn't happen if your carry pointers are
                    // always list pointers.
                    Val::Leaf(_) => {
                        self.current = None;
                        None
                    }
                }
            }
            Val::List(ix, list) => {
                match list[ix as usize] {
                    Res(0) => {
                        self.current = None;
                        None
                    }

                    other => {
                        self.current = Some(Val::List((ix + 1) as u8, list));
                        Some(self.runtime.fetch(other))
                    }
                }
                /*let offset = ix as usize;
                match list[offset] {
                    Res(0) => None,
                    other => {
                        self.current = Some(Val::List(ix + 1, list));
                        Some(self.runtime.fetch(other))
                    }
                }*/
            }
        }
        /*self.current = self.runtime.next(current);
        Some(current)*/
    }
    pub fn int(&mut self, fieldname: &'static str) -> Result<i32, TypeError> {
        match self.next().ok_or(TypeError::ArgumentUnderflow)? {
            Val::Leaf(Leaf::Int(i)) => Ok(i),
            Val::Leaf(leaf) => Err(TypeError::UnexpectedType {
                fieldname,
                expected: "int",
                got: leaf.typename(),
            }),
            Val::List(_, _) => todo!(),
        }
    }
}
pub enum Stack {
    Lambda,
}
pub enum Computation {
    List(Vec<Self>),
    Leaf(Leaf),
}
pub fn load_lisp() -> anyhow::Result<()> {
    let inp = ListInput::new(include_str!("maths_prog"))?;

    let (mut lisp, entrypoint) = Lisp::new(inp);
    println!("{:?}", &lisp.leafs[0..32]);
    println!("{:?}", &lisp.lists[0..16]);
    println!("Entrypoints: {:?}", entrypoint);
    lisp.eval(Res(-1));
    Ok(())
}
pub struct ListProc {
    arena: Vec<Interior>,

    freelist: FreeMap,
    strings: Rodeo,
    keywords: Rodeo,
    //tensors: Vec<*mut ggml_tensor>,
}

impl ListProc {
    pub fn new(keywords: Rodeo) -> Self {
        Self {
            arena: vec![Interior::Nil; 8096],
            freelist: FreeMap::new(8096),
            strings: Rodeo::new(),
            keywords,
        }
    }
    pub fn alloc(&mut self, interior: Interior) -> Operator {
        let ptr = self.freelist.allocate();
        self.arena[ptr] = interior;
        Operator {
            ptr: ptr as u32,
            operation: interior,
        }
    }
    fn keyword(&mut self, kw: Spur) -> Lambda {
        match self.keywords.resolve(&kw) {
            "eq" => Lambda::Eq,
            "if" => Lambda::Eq,
            _ => Lambda::Anon(kw),
        }
    }
    pub fn trace_roots(&mut self, ptr: u32, hs: &mut HashSet<u32>) {
        let v = self.arena[ptr as usize];
        hs.insert(ptr);
        match v {
            Interior::Ptr(r0) => {
                self.trace_roots(r0, hs);
            }

            Interior::Cons(r1, r2) => {
                self.trace_roots(r1, hs);
                self.trace_roots(r2, hs);
            }
            _ => (),
        };
    }
    pub fn collect_garbage(&mut self) {
        let mut active = HashSet::with_capacity(8096);
        self.trace_roots(0, &mut active);

        /*for cell in (0..self.freelist.max()) {
            let v = self.arena[cell];
            match v {
                Interior::Ptr(r0) => {active.insert(r0);},

                Interior::Cons(r1, r2) => {

                    active.insert(r1);
                    active.insert(r2);

                },
               _ => (),
            };
        }*/
    }
    pub fn eval(&mut self, list: List) -> Operator {
        /*let i = match list {
            List::Bool(b) => Interior::Bool(b),
            List::Int(i) => Interior::Int(i),
            List::Float(f) => Interior::Float(f.to_ne_bytes()),
            List::String(s) => Interior::String(self.strings.get_or_intern(s)),
            List::Keyword(spur) => Interior::Lambda(self.keyword(spur)),
            List::List(lists) => {
                for list in lists {
                    let interior = self.eval(list);
                    interior.front(cdr)
                }
                /*while let Some(next) = iter.next() {

                    let ev = self.eval(next)
                }*/
                Interior::Nil
            }
            List::Nil => Interior::Nil,
        };*/
        //self.alloc(i)
        todo!()
    }
}
pub enum TypeError {
    NonList,
    ArgumentUnderflow,
    UnexpectedType {
        fieldname: &'static str,
        expected: &'static str,
        got: &'static str,
    },
}
pub fn stupid_lisp() -> anyhow::Result<()> {
    let mut parser = Parser::new(include_str!("maths_prog"))?;
    let program = parser.parse_all()?;
    let interner = parser.interner;
    let mut runtime = ListProc::new(interner);
    for list in program {
        let res = runtime.eval(list);
        println!("{res:?}");
    }

    Ok(())
}
#[derive(Debug)]
pub struct Operator {
    ptr: u32,
    operation: Interior,
}
impl Operator {
    pub fn value(&self) -> Interior {
        self.operation
    }
    pub fn front(&self, cdr: &Operator) -> Interior {
        Interior::Cons(self.ptr, cdr.ptr)
    }
    pub fn back(&self, car: &Operator) -> Interior {
        Interior::Cons(car.ptr, self.ptr)
    }
    pub fn ptr(&self) -> Interior {
        Interior::Ptr(self.ptr)
    }
    pub fn cdr(&self) -> Result<Interior, TypeError> {
        match self.operation {
            Interior::Cons(_car, cdr) => Ok(Interior::Ptr(cdr)),
            Interior::Nil => Ok(Nil),
            _ => Err(TypeError::NonList),
        }
    }
    pub fn car(&self) -> Result<Interior, TypeError> {
        match self.operation {
            Interior::Cons(_car, cdr) => Ok(Interior::Ptr(cdr)),
            Interior::Nil => Ok(Nil),
            _ => Err(TypeError::NonList),
        }
    }
}
//<const PRECISION: usize>
/*#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct EqFloat(f64);
impl Eq for EqFloat {}*/
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq)]
pub enum Interior {
    Ptr(u32),
    Float([u8; 4]),
    Bool(bool),
    Int(i64),
    Tensor(*mut ggml_tensor),
    Cons(u32, u32),
    Lambda(Lambda),
    String(Spur),
    Nil,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq)]
pub enum Lambda {
    Anon(Spur),
    Let,
    If,
    Not,
    True,  //(x,y)->x
    False, //(x,y)->y
    Add,   //(x,y)->z
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
}
