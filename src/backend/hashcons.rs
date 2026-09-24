use std::{
    collections::HashMap,
    ffi::c_void,
    fmt::{Debug, Write},
    ops::{Index, IndexMut},
    ptr::null_mut,
};
pub const LAMBDA: &str = "λ";
use lasso::{Key, Rodeo, Spur};

use crate::runtime::memory::FreeMap;
#[repr(u8)]
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Cons,
    Nil,
    True,
    False,
    Atom,
    String,
    Lambda,
}
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct Cell {
    ty: Type,
    ptr: [u8; 3],
}
impl Cell {
    pub fn ptr(&self) -> usize {
        let mut data = [0u8; std::mem::size_of::<usize>()];
        data[0..3].copy_from_slice(&self.ptr);
        usize::from_ne_bytes(data)
    }
}
pub struct Reference<'a> {
    rt: &'a HashedLisp,
    cell: Cell,
}
impl<'a> std::fmt::Debug for Reference<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cell.ty {
            Type::Int => {
                let val = unsafe { self.rt.cells[self.cell].int };
                val.fmt(f)
            }
            Type::Float => {
                let float = unsafe { self.rt.cells[self.cell].float };
                float.fmt(f)
            }

            Type::Nil => f.write_str("Nil"),
            Type::True => f.write_str("true"),
            Type::False => f.write_str("false"),
            Type::Atom =>
            /*self.rt.atoms.resolve(&self.cell.spur()).fmt(f)*/
            {
                if let Some(spur) = self.cell.spur() {
                    let s = self.rt.atoms.resolve(&spur);
                    f.write_str(s)
                } else {
                    f.write_str("<INVALID ATOM>")
                }
            }
            Type::String => {
                if let Some(spur) = self.cell.spur() {
                    let s = self.rt.strings.resolve(&spur);
                    f.write_char('"')?;
                    f.write_str(s)?;
                    f.write_char('"')
                } else {
                    f.write_str("<INVALID STRING>")
                }
            }
            Type::Cons => {
                let Cons { car, cdr } = self.rt.conses[self.cell];
                let car = Reference {
                    rt: self.rt,
                    cell: car,
                };
                let cdr = Reference {
                    rt: self.rt,
                    cell: cdr,
                };
                write!(f, "({car} {cdr})")
            }
            Type::Lambda => {
                let val = unsafe { self.rt.cells[self.cell].pointer };
                write!(f, "λ{val:?}")
            }
        }
    }
}
impl<'a> std::fmt::Display for Reference<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cell.ty {
            Type::Int => {
                //Safe because the type has been checked
                let val = unsafe { self.rt.cells[self.cell].int };
                write!(f, "{val}")
            }
            Type::Float => {
                //Safe because the type has been checked
                let val = unsafe { self.rt.cells[self.cell].float };
                write!(f, "{val}")
            }

            Type::Nil => f.write_str("Nil"),
            Type::True => f.write_str("true"),
            Type::False => f.write_str("false"),
            Type::Atom =>
            /*self.rt.atoms.resolve(&self.cell.spur()).fmt(f)*/
            {
                if let Some(spur) = self.cell.spur() {
                    let s = self.rt.atoms.resolve(&spur);
                    f.write_str(s)
                } else {
                    f.write_str("<INVALID ATOM>")
                }
            }
            Type::String => {
                if let Some(spur) = self.cell.spur() {
                    let s = self.rt.strings.resolve(&spur);
                    f.write_char('"')?;
                    f.write_str(s)?;
                    f.write_char('"')
                } else {
                    f.write_str("<INVALID STRING>")
                }
            }
            Type::Cons => {
                let mut iterator = self.rt.iter_cell(self.cell);
                f.write_char('(')?;
                if let Some(first) = iterator.next() {
                    let re = Reference {
                        rt: self.rt,
                        cell: first,
                    };
                    write!(f, "{re}")?;
                }
                while let Some(next) = iterator.next() {
                    let re = Reference {
                        rt: self.rt,
                        cell: next,
                    };
                    write!(f, " {re}")?;
                    //next.fmt(f)?;
                }
                f.write_char(')')
            }
            Type::Lambda => {
                //Safe because only reading a number
                let val = unsafe { self.rt.cells[self.cell].pointer };
                write!(f, "λ{val:?}")
            }
        }
    }
}
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct Decimal([u8; 8]);
impl From<f64> for Decimal {
    #[inline]
    fn from(value: f64) -> Self {
        Decimal(value.to_ne_bytes())
    }
}
impl From<f32> for Decimal {
    #[inline]
    fn from(value: f32) -> Self {
        Decimal((value as f64).to_ne_bytes())
    }
}
impl From<Decimal> for f64 {
    #[inline]
    fn from(value: Decimal) -> Self {
        value.fp64()
    }
}
impl From<Decimal> for f32 {
    #[inline]
    fn from(value: Decimal) -> Self {
        value.fp32()
    }
}
impl Decimal {
    #[inline]
    pub fn fp32(&self) -> f32 {
        self.fp64() as f32
    }
    #[inline]
    pub fn fp64(&self) -> f64 {
        f64::from_ne_bytes(self.0)
    }
}
impl std::fmt::Debug for Decimal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.fp64())
    }
}
impl std::fmt::Display for Decimal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}", self.fp64())
    }
}

pub enum VMValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum Value {
    Keyword(String),
    String(String),
    Int(i64),
    Float(Decimal),
    Bool(bool),
    List(Vec<Self>),
    Cell(Cell),
    Lambda(Lambda),
    Nil,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct Cons {
    car: Cell,
    cdr: Cell,
}
impl Cons {
    pub fn empty() -> Self {
        Self {
            car: Cell::NIL,
            cdr: Cell::NIL,
        }
    }
}
impl Cell {
    pub const NIL: Self = Cell {
        ty: Type::Nil,
        ptr: [0u8; 3],
    };
    pub fn new(ty: Type, arr_ptr: usize) -> Self {
        let mut ptr = [0u8; 3];
        ptr.copy_from_slice(&arr_ptr.to_ne_bytes()[0..3]);
        Self { ty, ptr }
    }
    pub fn text(ty: Type, spur: Spur) -> Self {
        let mut ptr = [0u8; 3];
        ptr.copy_from_slice(&spur.into_inner().get().to_be_bytes()[0..3]);
        Self { ty, ptr }
    }
    pub fn spur(&self) -> Option<Spur> {
        Spur::try_from_usize(self.ptr())
    }
}

pub struct CellBuilder<'a> {
    runtime: &'a mut HashedLisp,
}

impl<'a> CellBuilder<'a> {
    pub fn register_value(&mut self, value: &Value) -> Cell {
        self.runtime.register_value(value)
    }
    pub fn cache(&mut self, value: Value, cell: Cell) {
        self.runtime.outsider.insert(value, cell);
    }
    pub fn int(&mut self, int: impl Into<i64>) -> Cell {
        let i: i64 = int.into();
        let pointee = StoredCell { int: i };
        let ptr = self.runtime.cells.alloc_new(pointee);
        let c = Cell::new(Type::Int, ptr);
        self.cache(Value::Int(i), c);
        c
    }
    pub fn float(&mut self, int: impl Into<f64>) -> Cell {
        let i: f64 = int.into();
        let pointee = StoredCell { float: i };
        let ptr = self.runtime.cells.alloc_new(pointee);
        Cell::new(Type::Float, ptr)
    }
    pub fn nil(&mut self) -> Cell {
        Cell {
            ty: Type::Nil,
            ptr: [0u8; 3],
        }
    }
    pub fn bool(&mut self, bool: bool) -> Cell {
        if bool {
            Cell {
                ty: Type::True,
                ptr: [0u8; 3],
            }
        } else {
            Cell {
                ty: Type::False,
                ptr: [0u8; 3],
            }
        }
    }
    pub fn atom(&mut self, kw: impl AsRef<str>) -> Cell {
        let kw = self.runtime.atoms.get_or_intern(kw);
        Cell::text(Type::Atom, kw)
    }
    pub fn lambda(&mut self, lambda: Lambda) -> Cell {
        let fnp = lambda.clos as *mut c_void;
        let pointee = StoredCell { pointer: fnp };
        let ptr = self.runtime.cells.alloc_new(pointee);
        Cell::new(Type::Float, ptr)
    }
    pub fn string(&mut self, s: impl AsRef<str>) -> Cell {
        let s = self.runtime.strings.get_or_intern(s);
        Cell::text(Type::String, s)
    }
    pub fn _cons_no_cache(&mut self, car: Cell, cdr: Cell) -> Cell {
        let ptr = self.runtime.conses.alloc_new(Cons { car, cdr });
        Cell::new(Type::Cons, ptr)
    }
    pub fn cons(&mut self, car: Cell, cdr: Cell) -> Cell {
        let key = Cons { car, cdr };
        if let Some(&cell) = self.runtime.cons_cache.get(&key) {
            return cell;
        }
        let ptr = self.runtime.conses.alloc_new(key);
        let cell = Cell::new(Type::Cons, ptr);
        self.runtime.cons_cache.insert(key, cell);
        cell
    }

    fn _fetch_cons(&self, ptr: Cell) -> Cons {
        self.runtime.conses[ptr]
    }
    fn _set_car(&mut self, cons: Cell, car: Cell) {
        let cons = &mut self.runtime.conses[cons];
        cons.car = car;
    }
    fn _set_cdr(&mut self, cons: Cell, cdr: Cell) {
        let cons = &mut self.runtime.conses[cons];
        cons.cdr = cdr;
    }
}
impl From<i64> for StoredCell {
    fn from(value: i64) -> Self {
        StoredCell { int: value }
    }
}

impl From<Lambda> for StoredCell {
    fn from(value: Lambda) -> Self {
        (value.clos as *mut c_void).into()
    }
}
impl From<f64> for StoredCell {
    fn from(value: f64) -> Self {
        StoredCell { float: value }
    }
}
impl<T> From<*mut T> for StoredCell {
    fn from(value: *mut T) -> Self {
        StoredCell {
            pointer: value as *mut c_void,
        }
    }
}
impl<T> From<*const T> for StoredCell {
    fn from(value: *const T) -> Self {
        StoredCell {
            pointer: value as *mut c_void,
        }
    }
}
pub union StoredCell {
    int: i64,
    float: f64,
    pointer: *mut c_void,
}
impl std::fmt::Debug for StoredCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { self.pointer.fmt(f) }
    }
}
impl StoredCell {
    pub fn empty() -> Self {
        Self {
            pointer: null_mut(),
        }
    }
    pub unsafe fn as_pointer<T>(&self) -> *mut T {
        let p = unsafe { self.pointer };
        p as *mut T
    }
    pub unsafe fn as_lambda(&self) -> Lambda {
        let f: fn(LispEvaluator<'_>) -> Result<Value, Cell> =
            unsafe { std::mem::transmute(self.pointer) };
        Lambda { clos: f }
    }
}
pub struct SlotMap<T: Debug> {
    free: FreeMap,
    data: Vec<T>,
}
impl<T: Debug> IndexMut<Cell> for SlotMap<T> {
    fn index_mut(&mut self, index: Cell) -> &mut Self::Output {
        self.data.index_mut(index.ptr())
    }
}
impl<T: Debug> Index<Cell> for SlotMap<T> {
    type Output = T;

    fn index(&self, index: Cell) -> &Self::Output {
        &self.data[index.ptr()]
    }
}
impl<T: Debug> Index<usize> for SlotMap<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T: Debug> SlotMap<T> {
    pub fn new(size: usize, empty: fn() -> T) -> Self {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(empty());
        }
        Self {
            free: FreeMap::new(size),
            data,
        }
    }
    fn alloc_new(&mut self, data: T) -> usize {
        let ptr = self.free.allocate();
        println!("allocating {data:?} into {ptr}");
        self.data[ptr] = data;
        ptr
    }
}
pub struct HashedLisp {
    cells: SlotMap<StoredCell>,
    conses: SlotMap<Cons>,
    strings: Rodeo,
    atoms: Rodeo,
    outsider: HashMap<Value, Cell>,
    cons_cache: HashMap<Cons, Cell>,
}
impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}
impl From<&'static str> for Value {
    fn from(value: &'static str) -> Self {
        Self::Keyword(value.into())
    }
}
impl<const N: usize> From<[Value; N]> for Value {
    fn from(value: [Value; N]) -> Self {
        Self::List(value.into())
    }
}
pub fn hashed_lisp() {
    let example_program: Value = [
        "+".into(),
        ["+".into(), 1i64.into(), 1i64.into()].into(),
        ["+".into(), 1i64.into(), 1i64.into()].into(),
        ["+".into(), 1i64.into(), 1i64.into()].into(),
        1i64.into(),
    ]
    .into();
    let mut lisp = HashedLisp::new();
    let cell = lisp.register_value(&example_program);
    //let mut iter = lisp.iter_cell(cell);
    let re = Reference { rt: &lisp, cell };
    println!("{re}");
    let result = lisp.eval(cell);

    println!("{result}");
    lisp.dump_cache();
    let result = lisp.eval(cell);
    println!("{result}");
    //lisp.dump_cache();
    //let conses = &lisp.conses.data[0..10];
    //println!("{conses:?}");
    /*while let Some(c) = iter.next() {
        let r = Reference { rt: &lisp, cell: c };
        println!("{r}");
    }*/
}
pub struct LispIterator<'a> {
    runtime: &'a HashedLisp,
    placeholder: Option<Cons>,
}
pub struct LispEvaluator<'a> {
    runtime: &'a mut HashedLisp,
    placeholder: Option<Cons>,
}
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct Lambda {
    clos: fn(LispEvaluator<'_>) -> Result<Value, Cell>,
    // dat: Cell,
}
impl Lambda {
    pub fn execute(&self, eval: LispEvaluator<'_>) -> Result<Value, Cell> {
        (self.clos)(eval)
    }
}
pub struct FnTable {
    //storage: HashMap<Spur, fn(LispEvaluator<'_>) -> Value>,
}
pub fn dispatch(fname: &str) -> fn(LispEvaluator<'_>) -> Result<Value, Cell> {
    match fname {
        "+" => integer_sum,
        "-" => integer_subtraction,
        _ => yield_nonexistent_fn,
    }
}

fn yield_nonexistent_fn(eval: LispEvaluator<'_>) -> Result<Value, Cell> {
    Err(eval
        .runtime
        .register_value(&Value::String("Function doesnt exist".into())))
}
fn integer_subtraction(mut eval: LispEvaluator<'_>) -> Result<Value, Cell> {
    let mut output: i64 = 0;
    if let Some(int) = eval.get_int()? {
        output += int;
    } else {
        return Ok(Value::Int(output));
    }
    loop {
        if let Some(int) = eval.get_int()? {
            output -= int;
        } else {
            return Ok(Value::Int(output));
        }
    }
}
fn integer_sum(mut eval: LispEvaluator<'_>) -> Result<Value, Cell> {
    let mut output: i64 = 0;
    loop {
        if let Some(int) = eval.get_int()? {
            output += int;
        } else {
            return Ok(Value::Int(output));
        }
    }
}
/*impl FnTable {
    fn run(atom: Spur) ->  {
        todo!()
    }
}*/
impl<'a> LispEvaluator<'a> {
    pub fn get_int(&mut self) -> Result<Option<i64>, Cell> {
        if let Some(n) = self.next() {
            let n = self.runtime.eval(n).cell;
            if n.ty == Type::Int {
                unsafe { Ok(Some(self.runtime.cells[n].int)) }
            } else {
                Err(n)
            }
        } else {
            Ok(None)
        }
    }
    pub fn next(&mut self) -> Option<Cell> {
        let Cons { car, cdr } = self.placeholder?;
        self.get_cons(cdr);

        Some(car)
    }
    pub fn add(mut self) -> Result<i64, Cell> {
        let mut output: i64 = 0;
        loop {
            if let Some(int) = self.get_int()? {
                output += int;
            } else {
                return Ok(output);
            }
        }
    }
    fn get_cons(&mut self, cell: Cell) {
        self.placeholder = if cell.ty == Type::Cons {
            Some(self.runtime.conses[cell])
        } else {
            None
        };
    }
}
impl<'a> LispIterator<'a> {
    fn get_cons(&mut self, cell: Cell) {
        self.placeholder = if cell.ty == Type::Cons {
            Some(self.runtime.conses[cell])
        } else {
            None
        };
    }
    pub fn get_int(&mut self) -> Result<i64, Option<Cell>> {
        if let Some(n) = self.next() {
            if n.ty == Type::Int {
                unsafe { Ok(self.runtime.cells[n].int) }
            } else {
                Err(Some(n))
            }
        } else {
            Err(None)
        }
    }
    pub fn next(&mut self) -> Option<Cell> {
        let Cons { car, cdr } = self.placeholder?;
        self.get_cons(cdr);

        Some(car)
    }
}
impl HashedLisp {
    pub fn dump_cache(&self) {
        println!("Cache Dump ================================");
        for (k, v) in &self.outsider {
            print!("{k:?} -> {v:?} | ");
            if let Value::Cell(c) = k {
                let k = Reference { rt: self, cell: *c };
                let v = Reference { rt: self, cell: *v };
                println!("{k} -> {v}")
            } else {
                let v = Reference { rt: self, cell: *v };
                println!("{k:?} -> {v}")
            }
            println!("")
        }
        println!("==========================================");
    }
    pub fn new() -> Self {
        Self {
            cells: SlotMap::new(8096, StoredCell::empty),
            conses: SlotMap::new(8096, Cons::empty),
            strings: Rodeo::new(),
            atoms: Rodeo::new(),
            outsider: HashMap::new(),
            cons_cache: HashMap::new(),
        }
    }

    pub fn eval(&mut self, cell: Cell) -> Reference<'_> {
        let key = Value::Cell(cell);
        if let Some(cached) = self.outsider.get(&key) {
            let result = Reference {
                rt: self,
                cell: *cached,
            };
            let start = Reference { rt: self, cell };

            println!("CACHE HIT:  {start} -> {result}");
            return result;
        }
        let start = Reference { rt: self, cell };
        //println!("CACHE MISS: {cell:?}", );
        println!("CACHE MISS: {start}",);
        drop(start);
        let result = match cell.ty {
            Type::Cons => self.eval_cons(cell).cell,
            _ => cell,
        };
        let start = Reference { rt: self, cell };
        println!(
            "CACHED {start} -> {}",
            Reference {
                rt: start.rt,
                cell: result
            }
        );
        self.outsider.insert(Value::Cell(start.cell), result);
        Reference {
            rt: self,
            cell: result,
        }
    }

    pub fn eval_cons(&mut self, cons: Cell) -> Reference<'_> {
        let mut iterator = self.iter_cell(cons);
        let result = if let Some(Cell {
            ty: Type::Atom,
            ptr,
        }) = iterator.next()
        {
            let aname = Cell {
                ty: Type::Atom,
                ptr,
            }
            .spur()
            .expect("expected atom");
            let name = self.atoms.resolve(&aname);
            let function = dispatch(&name);
            let p = iterator.placeholder;
            drop(iterator);
            let evaluator = LispEvaluator {
                runtime: self,
                placeholder: p,
            };
            let value = function(evaluator).unwrap();
            let cell = self.register_value(&value);
            Reference { rt: self, cell }
        } else {
            Reference {
                rt: self,
                cell: cons,
            }
        };

        result
    }
    pub fn iter_cell(&'_ self, cell: Cell) -> LispIterator<'_> {
        let mut l = LispIterator {
            runtime: self,
            placeholder: None,
        };
        l.get_cons(cell);
        l
    }
    pub fn register_value(&mut self, value: &Value) -> Cell {
        if let Some(cached) = self.outsider.get(value) {
            return *cached;
        }
        let is_list = if let Value::List(_) = value {
            true
        } else {
            false
        };
        let mut builder = CellBuilder { runtime: self };

        let result = match value {
            Value::Int(i) => builder.int(*i),
            Value::Bool(b) => builder.bool(*b),
            Value::List(values) => {
                let mut cell = Cell::NIL;
                for value in values.iter().rev() {
                    let value_cell = builder.register_value(value);
                    cell = builder.cons(value_cell, cell);
                }
                if values.is_empty() { Cell::NIL } else { cell }
            }
            Value::Nil => builder.nil(),
            Value::Keyword(kw) => builder.atom(kw),
            Value::String(s) => builder.string(s),
            Value::Cell(cell) => *cell,
            Value::Lambda(_lambda) => todo!(),
            Value::Float(decimal) => builder.float(*decimal),
        };
        if !is_list {
            self.outsider.insert(value.clone(), result);
        }
        result
    }
}
