//! The one Chez Scheme of the process, on its own thread, with the Tree libraries and `evolve.ss`.
//!
//! Agents get REPL sessions in it ([`SchemeHandle::run`]) and the evaluator runs candidate passes in
//! it ([`SchemeHandle::evaluate`]). Chez allows one instance per process, so every caller goes
//! through this thread.
use std::sync::mpsc::{Sender, channel};
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, anyhow, bail};
use chez::{Scheme, Value};

const LIBS: &[(&str, &str)] = &[
    ("tree spirv", include_str!("../../tensorlisp/src/scheme/stdlib/tree_spirv.ss")),
    ("tree", include_str!("../../tensorlisp/src/scheme/stdlib/tree.ss")),
    ("tree opt", include_str!("../../tensorlisp/src/scheme/stdlib/tree_opt.ss")),
    ("tree kernels", include_str!("../../tensorlisp/src/scheme/stdlib/tree_kernels.ss")),
    ("tree ptx", include_str!("../../tensorlisp/src/scheme/stdlib/tree_ptx.ss")),
    ("tree msl", include_str!("../../tensorlisp/src/scheme/stdlib/tree_msl.ss")),
    ("evolve", include_str!("evolve.ss")),
];

type Job = Box<dyn FnOnce(&Scheme) + Send>;

pub struct SchemeHandle {
    jobs: Mutex<Sender<Job>>,
}

static HANDLE: OnceLock<std::result::Result<SchemeHandle, String>> = OnceLock::new();

/// One row of an evaluation: a generator expression compiled with the baseline and the candidate pipeline.
#[derive(Clone)]
pub struct Row {
    pub expr: String,
    /// `ok`, `error` (compile failed) or `structure` (the interface or effects changed).
    pub status: String,
    pub note: String,
    pub base_size: i64,
    pub cand_size: i64,
    /// The candidate's SPIR-V, for the validator.
    pub bytes: Option<Vec<u8>>,
    /// The baseline pipeline's SPIR-V: a candidate is only blamed for what the baseline does not also do.
    pub base_bytes: Option<Vec<u8>>,
    /// PTX / MSL targets: the compile replies (code, entry point, interface) of the candidate and the baseline.
    pub cand_code: Option<Compiled>,
    pub base_code: Option<Compiled>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Compiled {
    pub text: String,
    pub entry: String,
    /// Per binding: `ro`, `wo` or `rw`.
    pub accesses: Vec<String>,
    pub push_bytes: u32,
    pub workgroup: [u32; 3],
}

fn compiled(v: &Value) -> Option<Compiled> {
    let Value::List(items) = v else { return None };
    let [Value::String(text), Value::String(entry), accesses, Value::Int(push), wg] = &items[..] else { return None };
    let accesses = match accesses {
        Value::List(a) => a.iter().map(|x| if let Value::Symbol(s) = x { s.clone() } else { "rw".into() }).collect(),
        _ => vec![],
    };
    let workgroup = [0, 1, 2].map(|i| match wg {
        Value::List(w) => match w.get(i) {
            Some(Value::Int(n)) => *n as u32,
            _ => 1,
        },
        _ => 1,
    });
    Some(Compiled { text: text.clone(), entry: entry.clone(), accesses, push_bytes: *push as u32, workgroup })
}

impl std::fmt::Debug for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Row({} {} {:?} {}->{} bytes={})", self.expr, self.status, self.note, self.base_size, self.cand_size, self.bytes.as_ref().map_or(0, |b| b.len()))
    }
}

#[derive(Debug, Clone)]
pub enum Outcome {
    /// The candidate's code (or its pipeline) did not load.
    LoadError(String),
    Rows(Vec<Row>),
}

pub fn handle() -> Result<&'static SchemeHandle> {
    HANDLE
        .get_or_init(|| {
            let (tx, rx) = channel::<Job>();
            let (ready_tx, ready_rx) = channel::<std::result::Result<(), String>>();
            let spawned = std::thread::Builder::new().name("evolve-scheme".into()).stack_size(256 << 20).spawn(move || {
                let scheme = match boot() {
                    Ok(s) => {
                        let _ = ready_tx.send(Ok(()));
                        s
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    job(&scheme);
                }
            });
            if let Err(e) = spawned {
                return Err(format!("cannot start the Scheme thread: {e}"));
            }
            match ready_rx.recv() {
                Ok(Ok(())) => Ok(SchemeHandle { jobs: Mutex::new(tx) }),
                Ok(Err(e)) => Err(e),
                Err(_) => Err("the Scheme thread died while starting".into()),
            }
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

fn boot() -> std::result::Result<Scheme, String> {
    let scheme = Scheme::new().map_err(|e| format!("Chez: {e}"))?;
    for (name, src) in LIBS {
        scheme.eval(src).map_err(|e| format!("loading (tl {name}): {e}"))?;
    }
    Ok(scheme)
}

fn as_string(v: &Value) -> String {
    match v {
        Value::String(s) | Value::Symbol(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

fn as_int(v: &Value) -> i64 {
    match v {
        Value::Int(i) => *i,
        _ => 0,
    }
}

impl SchemeHandle {
    pub fn on_thread<R: Send + 'static>(&self, f: impl FnOnce(&Scheme) -> R + Send + 'static) -> Result<R> {
        let (tx, rx) = channel();
        self.jobs
            .lock()
            .unwrap()
            .send(Box::new(move |s| {
                let _ = tx.send(f(s));
            }))
            .map_err(|_| anyhow!("the Scheme thread stopped"))?;
        rx.recv().map_err(|_| anyhow!("the Scheme thread stopped"))
    }

    /// Evaluates `code` in the named session (created on first use); returns what it printed and `=> value`.
    pub fn run(&self, session: &str, code: &str, timeout_ms: u64) -> Result<String> {
        let (session, code) = (session.to_string(), code.to_string());
        self.on_thread(move |s| {
            s.call("$evo-run", &[Value::String(session), Value::String(code), Value::Int(timeout_ms as i64)]).map(|v| as_string(&v)).map_err(|e| anyhow!("{e}"))
        })?
    }

    /// For each generator expression: does it build a kernel the passes cannot change (cooperative matrices)?
    /// An expression that fails to build counts as not locked.
    pub fn locked(&self, exprs: &[String]) -> Result<Vec<bool>> {
        let args = [Value::List(exprs.iter().map(|e| Value::String(e.clone())).collect())];
        let result = self.on_thread(move |s| s.call("$evo-locked", &args).map_err(|e| anyhow!("{e}")))??;
        match result {
            Value::List(items) => Ok(items.iter().map(|v| matches!(v, Value::Bool(true))).collect()),
            Value::Nil => Ok(Vec::new()),
            other => bail!("the Scheme side returned {other:?}"),
        }
    }

    pub fn reset(&self, session: &str) -> Result<()> {
        let session = session.to_string();
        self.on_thread(move |s| s.call("$evo-reset", &[Value::String(session)]).map(|_| ()).map_err(|e| anyhow!("{e}")))?
    }

    /// Compiles every expression with the default pipeline and with `pipeline` (after loading `code`).
    /// `generators`: the text of a candidate generator library named `(tl tree kernels-candidate)`, or "".
    pub fn evaluate(&self, code: &str, pipeline: &[String], target: &str, exprs: &[String], timeout_ms: u64, generators: &str) -> Result<Outcome> {
        let args = [
            Value::String(code.into()),
            Value::List(pipeline.iter().map(|p| Value::Symbol(p.clone())).collect()),
            Value::Symbol(target.into()),
            Value::List(exprs.iter().map(|e| Value::String(e.clone())).collect()),
            Value::Int(timeout_ms as i64),
            Value::String(generators.into()),
        ];
        let result = self.on_thread(move |s| s.call("$evo-evaluate", &args).map_err(|e| anyhow!("{e}")))??;
        let Value::List(items) = result else { bail!("the evaluator returned {result:?}") };
        match items.first() {
            Some(Value::Symbol(tag)) if tag == "load-error" => Ok(Outcome::LoadError(items.get(1).map(as_string).unwrap_or_default())),
            Some(Value::Symbol(tag)) if tag == "rows" => {
                let rows = items[1..]
                    .iter()
                    .map(|r| {
                        let Value::List(f) = r else { bail!("bad row {r:?}") };
                        if f.len() != 9 {
                            bail!("bad row {r:?}");
                        }
                        Ok(Row {
                            expr: as_string(&f[0]),
                            status: as_string(&f[1]),
                            note: as_string(&f[2]),
                            base_size: as_int(&f[3]),
                            cand_size: as_int(&f[4]),
                            bytes: if let Value::Bytevector(b) = &f[5] { Some(b.clone()) } else { None },
                            base_bytes: if let Value::Bytevector(b) = &f[6] { Some(b.clone()) } else { None },
                            cand_code: compiled(&f[7]),
                            base_code: compiled(&f[8]),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Outcome::Rows(rows))
            }
            _ => bail!("the evaluator returned {items:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = "(k:binary 'add)";

    #[test]
    fn sessions_keep_definitions_capture_output_and_time_out() {
        let h = handle().unwrap();
        assert_eq!(h.run("t1", "(define x 41)", 5000).unwrap(), "");
        assert_eq!(h.run("t1", "(+ x 1)", 5000).unwrap(), "=> 42");
        assert_eq!(h.run("t2", "x", 5000).unwrap().starts_with("error:"), true);
        assert_eq!(h.run("t1", "(display \"hi\") (newline) 'done", 5000).unwrap(), "hi\n=> done");
        assert!(h.run("t1", "(car 1)", 5000).unwrap().starts_with("error:"));
        let t = std::time::Instant::now();
        let out = h.run("t1", "(let loop () (loop))", 300).unwrap();
        assert!(out.contains("time limit"), "{out}");
        assert!(t.elapsed().as_secs() < 5);
        // the session survived the timeout
        assert_eq!(h.run("t1", "x", 5000).unwrap(), "=> 41");
        // an (import ...) is ignored with a note: the libraries are there already
        let out = h.run("t1", "(import (tl tree) (tl tree opt))\n(+ 1 2)", 5000).unwrap();
        assert!(out.contains("import ignored") && out.ends_with("=> 3"), "{out}");
        // no file access
        assert!(h.run("t1", "(open-input-file \"/etc/passwd\")", 5000).unwrap().starts_with("error:"));
        // the Tree helpers are there
        let out = h.run("t1", &format!("(tree-size {KERNEL})"), 5000).unwrap();
        assert!(out.starts_with("=> "), "{out}");
    }

    fn rows(code: &str, pipeline: &[&str]) -> Outcome {
        let p: Vec<String> = pipeline.iter().map(|s| s.to_string()).collect();
        handle().unwrap().evaluate(code, &p, "spirv", &[KERNEL.into(), "(k:rms-norm #f)".into()], 20_000, "").unwrap()
    }

    #[test]
    fn the_default_pipeline_scores_exactly_one() {
        let Outcome::Rows(rows) = rows("", &["fold", "licm", "cse", "dce"]) else { panic!() };
        assert_eq!(rows.len(), 2);
        for r in rows {
            assert_eq!(r.status, "ok", "{}: {}", r.expr, r.note);
            assert_eq!(r.base_size, r.cand_size);
            assert!(r.base_size > 0 && r.bytes.is_some());
        }
    }

    #[test]
    fn a_pass_that_drops_the_stores_is_caught_and_the_registry_is_restored() {
        let drop_stores = "(define (strip t) (cond [(and (pair? t) (eq? (car t) 'store)) '(begin)] [(pair? t) (map strip t)] [else t]))\n\
            (register-pass! 'drop-stores strip)";
        let Outcome::Rows(r) = rows(drop_stores, &["drop-stores"]) else { panic!() };
        assert!(r.iter().any(|r| r.status == "structure" && r.note.contains("stored to")), "{r:?}");
        // the sabotage pass is not left registered
        assert!(matches!(rows("", &["drop-stores"]), Outcome::LoadError(m) if m.contains("no pass named")));
    }

    #[test]
    fn load_errors_and_runaway_candidates_are_reported() {
        assert!(matches!(rows("(this is not scheme", &["fold"]), Outcome::LoadError(_)));
        assert!(matches!(rows("(car 5)", &["fold"]), Outcome::LoadError(_)));
        let spin = "(register-pass! 'spin (lambda (k) (let loop () (loop))))";
        let p: Vec<String> = vec!["spin".into()];
        let out = handle().unwrap().evaluate(spin, &p, "spirv", &[KERNEL.into()], 300, "").unwrap();
        let Outcome::Rows(r) = out else { panic!() };
        assert_eq!(r[0].status, "error");
        assert!(r[0].note.contains("time limit"), "{}", r[0].note);
    }

    #[test]
    fn kernels_with_cooperative_matrices_are_reported_as_locked() {
        let h = handle().unwrap();
        let exprs: Vec<String> = ["(k:binary 'add)", "(k:matmul-cm 'f16 32 32 16 1 1)", "(k:no-such-generator)"].iter().map(|s| s.to_string()).collect();
        assert_eq!(h.locked(&exprs).unwrap(), [false, true, false]);
        assert_eq!(h.locked(&[]).unwrap(), Vec::<bool>::new());
        let Outcome::Rows(rows) = h.evaluate("", &["fold".to_string()], "spirv", &exprs[..2], 20_000, "").unwrap() else { panic!() };
        assert_eq!((rows[0].note.as_str(), rows[1].note.as_str()), ("", "cmat"));
    }

    #[test]
    fn ptx_and_msl_targets_compile() {
        for target in ["ptx", "msl"] {
            let out = handle().unwrap().evaluate("", &["fold".to_string()], target, &[KERNEL.into()], 20_000, "").unwrap();
            let Outcome::Rows(r) = out else { panic!() };
            assert_eq!(r[0].status, "ok", "{target}: {}", r[0].note);
            assert!(r[0].cand_size > 100);
        }
    }
}
