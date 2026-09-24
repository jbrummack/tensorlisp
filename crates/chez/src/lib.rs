//! Embedded ChezScheme.
//!
//! [`Scheme`] owns the (process-global) Chez runtime. Values cross the boundary
//! as [`Value`], which is a Rust-owned copy: Chez's collector moves objects, so
//! raw `ptr`s are only held for the duration of a single call.
//!
//! Scheme errors are caught inside Scheme (see `prelude.ss`) and come back as
//! [`Error::Scheme`], so they never unwind through Rust frames.

pub mod sys;

use std::{
    ffi::{CString, c_void},
    marker::PhantomData,
    ptr::null,
    sync::atomic::{AtomicBool, Ordering},
};

use sys::ptr;

static PETITE_BOOT: &[u8] = include_bytes!(concat!(env!("CHEZ_BOOT_DIR"), "/petite.boot"));
static SCHEME_BOOT: &[u8] = include_bytes!(concat!(env!("CHEZ_BOOT_DIR"), "/scheme.boot"));
const PRELUDE: &str = include_str!("prelude.ss");

/// Chez keeps its heap in globals, so only one runtime can exist at a time.
static RUNNING: AtomicBool = AtomicBool::new(false);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("a Chez Scheme runtime is already running in this process")]
    AlreadyRunning,
    #[error("{0}")]
    Scheme(String),
    #[error("string contains a NUL byte: {0:?}")]
    Nul(String),
    #[error("{0} values cannot be passed to Scheme")]
    Unsupported(&'static str),
}

/// A Scheme value copied out of (or into) the Chez heap.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Void,
    Nil,
    Eof,
    Bool(bool),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
    Symbol(String),
    /// Proper list.
    List(Vec<Value>),
    /// Pair whose cdr is not a list.
    Pair(Box<Value>, Box<Value>),
    Vector(Vec<Value>),
    Bytevector(Vec<u8>),
    /// Something without a Rust representation (procedure, record, port,
    /// bignum out of i64 range, ...), named by kind.
    Opaque(&'static str),
}

impl Value {
    /// Copies a Chez object. Must not call back into Scheme, which could move
    /// objects that are still being walked.
    unsafe fn from_ptr(p: ptr) -> Value {
        unsafe {
            if sys::chez_fixnump(p) != 0 {
                Value::Int(sys::chez_fixnum_value(p) as i64)
            } else if sys::chez_nullp(p) != 0 {
                Value::Nil
            } else if p == sys::chez_void() {
                Value::Void
            } else if sys::chez_eof_objectp(p) != 0 {
                Value::Eof
            } else if sys::chez_booleanp(p) != 0 {
                Value::Bool(sys::chez_boolean_value(p) != 0)
            } else if sys::chez_flonump(p) != 0 {
                Value::Float(sys::chez_flonum_value(p))
            } else if sys::chez_charp(p) != 0 {
                Value::Char(char::from_u32(sys::chez_char_value(p)).unwrap_or('\u{FFFD}'))
            } else if sys::chez_stringp(p) != 0 {
                Value::String(string_from_ptr(p))
            } else if sys::chez_symbolp(p) != 0 {
                Value::Symbol(string_from_ptr(sys::Ssymbol_to_string(p)))
            } else if sys::chez_pairp(p) != 0 {
                list_from_ptr(p)
            } else if sys::chez_vectorp(p) != 0 {
                let n = sys::chez_vector_length(p);
                Value::Vector((0..n).map(|i| Value::from_ptr(sys::chez_vector_ref(p, i))).collect())
            } else if sys::chez_bytevectorp(p) != 0 {
                let n = sys::chez_bytevector_length(p) as usize;
                Value::Bytevector(std::slice::from_raw_parts(sys::chez_bytevector_data(p), n).to_vec())
            } else if sys::chez_bignump(p) != 0 {
                let mut v = 0i64;
                if sys::Stry_integer64_value(p, &mut v, std::ptr::null_mut()) != 0 {
                    Value::Int(v)
                } else {
                    Value::Opaque("bignum")
                }
            } else if sys::chez_procedurep(p) != 0 {
                Value::Opaque("procedure")
            } else {
                Value::Opaque("object")
            }
        }
    }

    /// Allocates the value in the Chez heap. Allocation from C never triggers
    /// a collection, so partially built structures stay valid.
    unsafe fn to_ptr(&self) -> Result<ptr, Error> {
        unsafe {
            Ok(match self {
                Value::Void => sys::chez_void(),
                Value::Nil => sys::chez_nil(),
                Value::Eof => sys::chez_eof_object(),
                Value::Bool(true) => sys::chez_true(),
                Value::Bool(false) => sys::chez_false(),
                Value::Int(i) => sys::Sinteger64(*i),
                Value::Float(f) => sys::Sflonum(*f),
                Value::Char(c) => sys::chez_char(*c as u32),
                Value::String(s) => sys::Sstring_utf8(s.as_ptr().cast(), s.len() as sys::iptr),
                Value::Symbol(s) => sys::Sstring_to_symbol(c_string(s)?.as_ptr()),
                Value::List(items) => {
                    let mut list = sys::chez_nil();
                    for item in items.iter().rev() {
                        list = sys::Scons(item.to_ptr()?, list);
                    }
                    list
                }
                Value::Pair(car, cdr) => sys::Scons(car.to_ptr()?, cdr.to_ptr()?),
                Value::Vector(items) => {
                    let v = sys::Smake_vector(items.len() as sys::iptr, sys::chez_false());
                    for (i, item) in items.iter().enumerate() {
                        sys::Svector_set(v, i as sys::iptr, item.to_ptr()?);
                    }
                    v
                }
                Value::Bytevector(bytes) => {
                    let bv = sys::Smake_bytevector(bytes.len() as sys::iptr, 0);
                    let data = sys::chez_bytevector_data(bv);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
                    bv
                }
                Value::Opaque(kind) => return Err(Error::Unsupported(kind)),
            })
        }
    }
}

unsafe fn string_from_ptr(p: ptr) -> String {
    unsafe {
        (0..sys::chez_string_length(p))
            .map(|i| char::from_u32(sys::chez_string_ref(p, i)).unwrap_or('\u{FFFD}'))
            .collect()
    }
}

unsafe fn list_from_ptr(mut p: ptr) -> Value {
    unsafe {
        let mut items = Vec::new();
        while sys::chez_pairp(p) != 0 {
            items.push(Value::from_ptr(sys::chez_car(p)));
            p = sys::chez_cdr(p);
        }
        if sys::chez_nullp(p) != 0 {
            return Value::List(items);
        }
        // Improper list: rebuild as nested pairs ending in the tail.
        items
            .into_iter()
            .rev()
            .fold(Value::from_ptr(p), |cdr, car| Value::Pair(Box::new(car), Box::new(cdr)))
    }
}

fn c_string(s: &str) -> Result<CString, Error> {
    CString::new(s).map_err(|_| Error::Nul(s.into()))
}

/// The embedded Chez runtime. Tied to the thread that created it.
pub struct Scheme {
    _not_send: PhantomData<*mut ()>,
}

impl Scheme {
    /// Boots Chez from the boot files embedded in this binary.
    pub fn new() -> Result<Self, Error> {
        if RUNNING.swap(true, Ordering::SeqCst) {
            return Err(Error::AlreadyRunning);
        }
        unsafe {
            sys::Sscheme_init(None);
            sys::Sregister_boot_file_bytes(
                c"petite".as_ptr(),
                PETITE_BOOT.as_ptr() as *mut c_void,
                PETITE_BOOT.len() as sys::iptr,
            );
            sys::Sregister_boot_file_bytes(
                c"scheme".as_ptr(),
                SCHEME_BOOT.as_ptr() as *mut c_void,
                SCHEME_BOOT.len() as sys::iptr,
            );
            sys::Sbuild_heap(null(), None);
        }
        let scheme = Scheme { _not_send: PhantomData };
        scheme.bootstrap_prelude();
        Ok(scheme)
    }

    fn bootstrap_prelude(&self) {
        // The prelude defines the error-catching helpers, so it is evaluated
        // with plain `eval`. Each step finishes before the next procedure is
        // looked up, since running Scheme may move objects.
        unsafe {
            let src = sys::Sstring_utf8(PRELUDE.as_ptr().cast(), PRELUDE.len() as sys::iptr);
            let port = sys::Scall1(self.top_level("open-input-string"), src);
            let form = sys::Scall1(self.top_level("read"), port);
            sys::Scall1(self.top_level("eval"), form);
        }
    }

    unsafe fn top_level(&self, name: &str) -> ptr {
        let name = c_string(name).expect("internal procedure name");
        unsafe { sys::Stop_level_value(sys::Sstring_to_symbol(name.as_ptr())) }
    }

    /// Unpacks the `(ok? . value)` pair returned by the prelude helpers.
    unsafe fn unwrap_result(result: ptr) -> Result<Value, Error> {
        unsafe {
            let value = Value::from_ptr(sys::chez_cdr(result));
            if sys::chez_boolean_value(sys::chez_car(result)) != 0 {
                Ok(value)
            } else if let Value::String(message) = value {
                Err(Error::Scheme(message))
            } else {
                Err(Error::Scheme(format!("{value:?}")))
            }
        }
    }

    /// Reads and evaluates every form in `src`, returning the last value.
    pub fn eval(&self, src: &str) -> Result<Value, Error> {
        unsafe {
            let src = sys::Sstring_utf8(src.as_ptr().cast(), src.len() as sys::iptr);
            let result = sys::Scall1(self.top_level("$tl-eval-string"), src);
            Self::unwrap_result(result)
        }
    }

    /// Calls the top-level procedure `name` with `args`.
    pub fn call(&self, name: &str, args: &[Value]) -> Result<Value, Error> {
        unsafe {
            let name = Value::Symbol(name.into()).to_ptr()?;
            let mut list = sys::chez_nil();
            for arg in args.iter().rev() {
                list = sys::Scons(arg.to_ptr()?, list);
            }
            let result = sys::Scall2(self.top_level("$tl-call"), name, list);
            Self::unwrap_result(result)
        }
    }

    /// Makes a C function visible to Scheme's `foreign-procedure` under
    /// `name`, e.g. `(foreign-procedure "ggml_add" (uptr uptr uptr) uptr)`.
    /// Registering the same name again replaces the entry.
    ///
    /// # Safety
    /// `addr` must be a C-ABI function that stays valid for the lifetime of
    /// the process, and every `foreign-procedure` declared for it must match
    /// its real signature.
    pub unsafe fn register_foreign(&self, name: &str, addr: *const c_void) -> Result<(), Error> {
        let name = c_string(name)?;
        unsafe { sys::Sregister_symbol(name.as_ptr(), addr as *mut c_void) };
        Ok(())
    }

    /// Runs the interactive Chez REPL on stdin/stdout until `(exit)`.
    pub fn repl(&self) -> i32 {
        unsafe {
            // Line editing; only takes effect when stdin is a terminal.
            sys::Senable_expeditor(null());
            sys::Sscheme_start(0, std::ptr::null_mut())
        }
    }
}

impl Drop for Scheme {
    fn drop(&mut self) {
        unsafe { sys::Sscheme_deinit() };
        RUNNING.store(false, Ordering::SeqCst);
    }
}
