//! `tl-evolve`: run agents that search for Tree optimization passes, evaluate candidates, and use the forum.
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use tensorlisp_evolve::agent::{self, AgentConfig};
use tensorlisp_evolve::{config, context};
use tensorlisp_evolve::embed::{CodeIndex, Embedder, HashEmbedder, HttpEmbedder};
use tensorlisp_evolve::eval::{self, EvalConfig, GateCommand};
use tensorlisp_evolve::forum::Forum;
use tensorlisp_evolve::kv::FsKv;
use tensorlisp_evolve::llm::Llm;
use tensorlisp_evolve::search::SearchIndex;
use tensorlisp_evolve::tools::ToolCtx;
use tensorlisp_evolve::weights;

#[derive(Parser)]
#[command(name = "tl-evolve", about = "Agents that search for Tree optimization passes, with a shared forum.")]
struct Cli {
    /// Repository root (agents read it; the workspace and index live under it).
    #[arg(long, global = true, default_value = ".")]
    repo: PathBuf,
    /// Where the forum lives: a directory used as a key-value store (put it on a shared filesystem to share it).
    #[arg(long, global = true, env = "TL_EVOLVE_STORE")]
    store: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct Common {
    /// spirv, ptx or msl.
    #[arg(long, default_value = "spirv")]
    target: String,
    /// Label for verdicts from device gates (what the gates ran on).
    #[arg(long, default_value = "static")]
    device: String,
    /// NAME=COMMAND, run with TL_TREE_PASS_FILE and TL_TREE_PASSES set; exit status is the verdict and
    /// `TL_EVOLVE_METRIC speedup=1.1` lines on stdout are recorded. Repeatable.
    #[arg(long = "gate")]
    gates: Vec<GateCommand>,
    /// Embedding model for similar_code (any OpenAI-compatible /embeddings endpoint). Default: a built-in structural embedding.
    #[arg(long, env = "TL_EVOLVE_EMBED_MODEL")]
    embed_model: Option<String>,
    #[arg(long, env = "TL_EVOLVE_AUTHOR")]
    author: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Run an agent.
    Run {
        /// Chat model (an OpenRouter id such as anthropic/claude-sonnet-4.5, or whatever your endpoint serves).
        #[arg(long, env = "TL_EVOLVE_MODEL")]
        model: String,
        #[command(flatten)]
        common: Common,
        /// What to do (default: look for a pipeline that beats the baseline).
        #[arg(long)]
        task: Option<String>,
        #[arg(long, default_value_t = 40)]
        turns: usize,
        /// The context window in tokens: `auto` asks the endpoint for the model's window (capped by
        /// --context-cap), or give a number (`200000`, `64k`).
        #[arg(long, default_value = "auto", env = "TL_EVOLVE_CONTEXT_TOKENS")]
        context_tokens: String,
        /// Upper bound for an automatically detected window: models with 1M-token windows are slower, costlier
        /// and worse long before they are full.
        #[arg(long, default_value_t = 200_000)]
        context_cap: usize,
        /// Compact when the conversation passes this share of the window: `0.75`, `75` or `75%`.
        #[arg(long, default_value = "75%", value_parser = context::parse_fraction)]
        compact_at: f64,
        /// Assistant turns kept verbatim when compacting.
        #[arg(long, default_value_t = 8)]
        keep_turns: usize,
        /// A separate (cheaper) model for the compaction summaries; default: the chat model.
        #[arg(long, env = "TL_EVOLVE_SUMMARY_MODEL")]
        summary_model: Option<String>,
        /// Continue this agent's previous run (its transcript and notes) instead of starting over.
        #[arg(long)]
        resume: bool,
        /// Show the agent how kernels are used in Rust: rust_map, rust_outline, launch_sites.
        #[arg(long)]
        rust: bool,
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long, short)]
        quiet: bool,
    },
    /// Evaluate a candidate file.
    Eval {
        file: PathBuf,
        /// Comma-separated pass names.
        #[arg(long)]
        pipeline: String,
        #[command(flatten)]
        common: Common,
        /// Kernel-expression substrings to report on separately (repeatable); the verdict is unchanged.
        #[arg(long)]
        focus: Vec<String>,
        /// A modified copy of tree_kernels.ss: kernels are built by it and compared with the original generators.
        #[arg(long)]
        generators: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Read and write the forum.
    Forum {
        #[command(subcommand)]
        action: ForumCmd,
        #[command(flatten)]
        common: Common,
    },
    /// Evaluate Scheme in the agent runtime (Tree libraries loaded).
    Scheme {
        code: String,
    },
    /// List the endpoint's models (OpenRouter and compatible): ids, context windows, free tier, tool support.
    Models {
        /// Only models with a free tier.
        #[arg(long)]
        free: bool,
        /// Only models that support tool calls (agents need them).
        #[arg(long)]
        tools: bool,
        /// Substring of the id.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Which kernels matter: profiles of real workloads, per device.
    Weights {
        #[command(subcommand)]
        action: WeightsCmd,
    },
    /// Measure a candidate on the device: run a command with the profiler on, with and without the candidate, and
    /// print the weighted per-kernel speedup and the run-to-run noise (as `TL_EVOLVE_METRIC` lines, so it works as
    /// a gate). The candidate is whatever TL_TREE_PASS_FILE / TL_TREE_PASSES / TL_TREE_KERNELS_FILE the environment
    /// holds, which is how `--gate` runs it.
    Bench {
        #[arg(long)]
        backend: String,
        /// Device label, to use that device's profile weights (default: the baseline's own time shares).
        #[arg(long)]
        device: Option<String>,
        /// Profiled runs for the baseline and for the candidate each (2 or more gives a noise estimate).
        #[arg(long, default_value_t = 2)]
        runs: usize,
        /// Leave out kernels launched fewer times than this in a run: a single cold launch can be ten times slower than
        /// a warm one.
        #[arg(long, default_value_t = 3)]
        min_launches: u64,
        /// Fail (exit 1) when the speedup is below this.
        #[arg(long)]
        min_speedup: Option<f64>,
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    /// The Rust implementation around the kernels (what `run --rust` shows agents).
    Impl {
        #[command(subcommand)]
        action: ImplCmd,
        #[command(flatten)]
        common: Common,
    },
    /// Existing kernels to learn from: ggml (Vulkan, CUDA, Metal) and the vLLM-derived paged attention.
    Ref {
        #[command(subcommand)]
        action: RefCmd,
        #[command(flatten)]
        common: Common,
    },
}

#[derive(Subcommand)]
enum WeightsCmd {
    /// Import a `TL_VULKAN_PROFILE=1` table (or lines `EXPR LAUNCHES BUSY_MS`); `-` reads stdin.
    Add {
        file: String,
        #[arg(long)]
        workload: String,
        /// Device label, matching the `--device` agents and gates use.
        #[arg(long)]
        device: String,
        /// Importance of this workload among the device's (default 1).
        #[arg(long, default_value_t = 1.0)]
        weight: f64,
    },
    /// Run a command with the backend's profiler on and import what it prints. Everything after `--` is the
    /// command, e.g. `weights capture --backend metal --device m1pro --workload tips -- cargo test --features native-metal-tree --test native_tips -- --nocapture`.
    Capture {
        #[arg(long)]
        backend: String,
        #[arg(long)]
        workload: String,
        #[arg(long)]
        device: String,
        #[arg(long, default_value_t = 1.0)]
        weight: f64,
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
    Show {
        #[arg(long)]
        device: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    Rm {
        #[arg(long)]
        workload: String,
        #[arg(long)]
        device: String,
    },
}

#[derive(Subcommand)]
enum ImplCmd {
    /// The files that decide how kernels are used, each with its own documentation.
    Map,
    /// Items of a .rs or .ss file with line numbers.
    Outline { file: String },
    /// Where a generator is defined and which Rust functions emit it.
    Sites { kernel: String },
    /// Which executors lower a graph op (omit the op for the whole matrix).
    Ops { op: Option<String> },
}

#[derive(Subcommand)]
enum RefCmd {
    /// Where ggml / vLLM implement the operation of a Tree generator expression.
    Kernels { kernel: String },
    Search {
        query: String,
        #[arg(long)]
        backend: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: u64,
    },
}

#[derive(Subcommand)]
enum ForumCmd {
    Search {
        query: String,
        #[arg(long)]
        field: Option<String>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: u64,
    },
    Grep {
        pattern: String,
        #[arg(long, short)]
        ignore_case: bool,
    },
    Read {
        id: String,
    },
    Leaderboard,
    Similar {
        /// File with code to look up.
        file: PathBuf,
    },
    Post {
        #[arg(long)]
        kind: String,
        #[arg(long)]
        headline: String,
        #[arg(long)]
        body: String,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long)]
        eval: Option<String>,
        #[arg(long)]
        parent: Option<String>,
    },
    Reply {
        id: String,
        body: String,
    },
}

fn embedder(model: Option<&str>) -> Arc<dyn Embedder> {
    match model {
        Some(m) => {
            let base = config::first_var(&["TL_EVOLVE_EMBED_BASE_URL", "TL_EVOLVE_BASE_URL", "OPENAI_BASE_URL"]).unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
            let key = config::first_var(&["TL_EVOLVE_EMBED_API_KEY", "TL_EVOLVE_API_KEY", "OPENROUTER_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]);
            Arc::new(HttpEmbedder::new(&base, key, m))
        }
        None => Arc::new(HashEmbedder),
    }
}

fn setup(cli: &Cli, common: &Common, default_author: &str) -> Result<ToolCtx> {
    let repo = cli.repo.canonicalize().with_context(|| format!("repository {}", cli.repo.display()))?;
    let store = cli.store.clone().unwrap_or_else(|| repo.join("evolve/forum"));
    let kv: Arc<dyn tensorlisp_evolve::kv::Kv> = Arc::new(FsKv::open(&store)?);
    let author = common.author.clone().unwrap_or_else(|| default_author.to_string());
    let safe: String = author.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' }).collect();
    let mut cfg = EvalConfig::new(&common.target, repo.join("evolve/work").join(&safe).join("eval"));
    cfg.device = common.device.clone();
    cfg.gates = common.gates.clone();
    let forum = Forum::new(kv.clone());
    let search = SearchIndex::in_dir(repo.join("evolve/index").join(&safe))?;
    let code = CodeIndex::new(embedder(common.embed_model.as_deref()), kv);
    ToolCtx::new(forum, search, code, cfg, repo.clone(), repo.join("evolve/work").join(&safe), &author)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Ok(repo) = cli.repo.canonicalize() {
        config::load_dotenv(&repo);
    }
    match &cli.command {
        Command::Models { free, tools, filter } => {
            let llm = Llm::from_env(Some("none"))?;
            let mut models = llm.models()?;
            models.retain(|m| (!*free || m.free) && (!*tools || m.tools) && filter.as_ref().is_none_or(|f| m.id.contains(f.as_str())));
            models.sort_by(|a, b| b.context_length.cmp(&a.context_length).then(a.id.cmp(&b.id)));
            for m in &models {
                println!("{:<56} {:>9} tokens  {}{}", m.id, m.context_length.map_or("?".to_string(), |c| c.to_string()), if m.free { "free " } else { "paid " }, if m.tools { "tools" } else { "no-tools" });
            }
            println!("{} models", models.len());
        }
        Command::Run { model, common, task, turns, context_tokens, context_cap, compact_at, keep_turns, summary_model, resume, rust, temperature, quiet } => {
            let default_author = format!("agent-{}", model.rsplit('/').next().unwrap_or(model));
            let mut ctx = setup(&cli, common, &default_author)?;
            ctx.rust_tools = *rust;
            let mut llm = Llm::from_env(Some(model))?;
            llm.temperature = *temperature;
            let summarizer = summary_model.as_deref().map(|m| Llm::from_env(Some(m))).transpose()?;
            let workspace = ctx.workspace.strip_prefix(&ctx.repo).unwrap_or(&ctx.workspace).display().to_string();
            let task = task.clone().unwrap_or_else(|| format!("Find a pass or pipeline that makes the Tree kernels smaller or faster than the baseline (fold licm cse dce) on {}. Start from the forum, and from `hot_kernels`.", common.target));
            let transcript = ctx.workspace.join("transcript.jsonl");
            let resume_messages = if *resume {
                let mut m = agent::replay(&transcript).with_context(|| "nothing to resume: run without --resume first")?;
                m.push(serde_json::json!({"role": "user", "content": "Your run was interrupted and has been resumed. Check your notes, then continue."}));
                Some(m)
            } else {
                None
            };
            let mut agent_cfg = AgentConfig::new(agent::system_prompt(&common.target, &common.device, &ctx.author, &workspace, *rust), task);
            agent_cfg.max_turns = *turns;
            agent_cfg.transcript = Some(transcript);
            let detected = if context_tokens.eq_ignore_ascii_case("auto") { llm.context_window() } else { None };
            let (window, why) = context::resolve_window(context_tokens, *context_cap, detected)?;
            eprintln!("context: {why}; compacting at {:.0}% = {} tokens", compact_at * 100.0, (window as f64 * compact_at) as usize);
            agent_cfg.context_tokens = window;
            agent_cfg.compact_at = *compact_at;
            agent_cfg.keep_turns = *keep_turns;
            agent_cfg.resume = resume_messages;
            agent_cfg.verbose = !*quiet;
            eprintln!("{} on {} via {}; workspace {workspace}{}", ctx.author, model, llm.base_url, if *resume { " (resumed)" } else { "" });
            let r = agent::run_with(&llm, summarizer.as_ref().map(|s| s as &dyn agent::Chat), &ctx, &agent_cfg)?;
            eprintln!("{} turns, {} tool calls, {} compactions, {} overflows recovered, {} prompt / {} completion tokens", r.turns, r.tool_calls, r.compactions, r.overflows, r.prompt_tokens, r.completion_tokens);
            println!("{}", r.summary.unwrap_or_else(|| "(the agent did not finish)".into()));
        }
        Command::Eval { file, pipeline, common, focus, generators, json } => {
            let ctx = setup(&cli, common, "cli")?;
            let code = std::fs::read_to_string(file).with_context(|| file.display().to_string())?;
            let pipeline: Vec<String> = pipeline.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect();
            let gen_text = generators.as_ref().map(|g| std::fs::read_to_string(g).with_context(|| g.display().to_string())).transpose()?;
            let ev = eval::evaluate_with(&ctx.forum, &ctx.eval_cfg_now()?, &code, &pipeline, gen_text.as_deref())?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&json!({"id": ev.id, "verdict": ev.verdict, "report": ev.report}))?);
            } else {
                print!("{}", eval::render_focus(&ev, focus));
            }
            if !ev.verdict.ok {
                std::process::exit(1);
            }
        }
        Command::Forum { action, common } => {
            let ctx = setup(&cli, common, &std::env::var("USER").unwrap_or_else(|_| "human".into()))?;
            let out = match action {
                ForumCmd::Search { query, field, kind, limit } => ctx.call("forum_search", &json!({"query": query, "field": field, "kind": kind, "limit": limit})),
                ForumCmd::Grep { pattern, ignore_case } => ctx.call("forum_grep", &json!({"pattern": pattern, "ignore_case": ignore_case})),
                ForumCmd::Read { id } => ctx.call("forum_read", &json!({"id": id})),
                ForumCmd::Leaderboard => ctx.call("forum_leaderboard", &json!({})),
                ForumCmd::Similar { file } => ctx.call("similar_code", &json!({"code": std::fs::read_to_string(file)?})),
                ForumCmd::Post { kind, headline, body, tags, eval, parent } => ctx.call("forum_post", &json!({"kind": kind, "headline": headline, "body": body, "tags": tags, "eval_id": eval, "parent": parent})),
                ForumCmd::Reply { id, body } => ctx.call("forum_reply", &json!({"id": id, "body": body})),
            };
            println!("{out}");
            if out.starts_with("error:") {
                std::process::exit(1);
            }
        }
        Command::Weights { action } => {
            let repo = cli.repo.canonicalize()?;
            let store = cli.store.clone().unwrap_or_else(|| repo.join("evolve/forum"));
            let kv = FsKv::open(&store)?;
            match action {
                WeightsCmd::Add { file, workload, device, weight } => {
                    let text = if file == "-" { std::io::read_to_string(std::io::stdin())? } else { std::fs::read_to_string(file).with_context(|| file.clone())? };
                    let w = weights::parse_profile(&text, workload, device, *weight)?;
                    weights::store(&kv, &w)?;
                    println!("stored {} Tree kernels for {device}/{workload} ({:.2} ms of {:.2} ms busy)", w.kernels.len(), w.kernels.iter().map(|k| k.busy_ms).sum::<f64>(), w.total_busy_ms);
                }
                WeightsCmd::Capture { backend, workload, device, weight, command } => {
                    // Metal can also record the exact launches (arguments and buffer sizes), which the kernel benchmark replays
                    let launches_file = std::env::temp_dir().join(format!("tl-evolve-launches-{}.jsonl", std::process::id()));
                    let _ = std::fs::remove_file(&launches_file);
                    let set: Vec<(&str, &str)> = if backend == "metal" { vec![("TL_METAL_LAUNCHES", launches_file.to_str().unwrap_or(""))] } else { vec![] };
                    let text = weights::capture_env(backend, &command.join(" "), &[], &set)?;
                    let w = weights::parse_profile(&text, workload, device, *weight)?;
                    weights::store(&kv, &w)?;
                    if let Ok(t) = std::fs::read_to_string(&launches_file) {
                        let launches = weights::parse_launches(&t);
                        weights::store_launches(&kv, device, workload, &launches)?;
                        println!("recorded {} distinct launches ({} total) for kernel benchmarks", launches.len(), launches.iter().map(|l| l.count).sum::<u64>());
                    }
                    let _ = std::fs::remove_file(&launches_file);
                    println!("captured {} Tree kernels for {device}/{workload} ({:.2} ms of {:.2} ms busy)", w.kernels.len(), w.kernels.iter().map(|k| k.busy_ms).sum::<f64>(), w.total_busy_ms);
                }
                WeightsCmd::Show { device, limit } => {
                    let profile = weights::load(&kv, device)?;
                    let exprs: Vec<String> = profile.weights.keys().cloned().collect();
                    let flags = if exprs.is_empty() { Vec::new() } else { tensorlisp_evolve::scheme::handle()?.locked(&exprs)? };
                    let locked: std::collections::HashSet<String> = exprs.into_iter().zip(flags).filter(|(_, f)| *f).map(|(e, _)| e).collect();
                    println!("{}", weights::render(&profile, device, *limit, &|e| locked.contains(e)));
                }
                WeightsCmd::Rm { workload, device } => {
                    weights::remove(&kv, device, workload)?;
                    println!("removed {device}/{workload}");
                }
            }
        }
        Command::Bench { backend, device, runs, min_launches, min_speedup, command } => {
            let cmd = command.join(" ");
            let w = match device {
                Some(d) => {
                    let repo = cli.repo.canonicalize()?;
                    let store = cli.store.clone().unwrap_or_else(|| repo.join("evolve/forum"));
                    weights::load(&FsKv::open(&store)?, d)?.weights
                }
                None => Default::default(),
            };
            let (mut base, mut cand) = (Vec::new(), Vec::new());
            for i in 0..(*runs).max(1) {
                eprintln!("bench: run {}/{}: baseline", i + 1, runs);
                base.push(tensorlisp_evolve::bench::profiled_run(backend, &cmd, true, &[])?);
                eprintln!("bench: run {}/{}: candidate", i + 1, runs);
                cand.push(tensorlisp_evolve::bench::profiled_run(backend, &cmd, false, &[])?);
            }
            let c = tensorlisp_evolve::bench::compare(&base, &cand, &w, *min_launches)?;
            print!("{}", tensorlisp_evolve::bench::report(&c));
            if min_speedup.is_some_and(|m| c.speedup < m) {
                std::process::exit(1);
            }
        }
        Command::Impl { action, common } => {
            let mut ctx = setup(&cli, common, "cli")?;
            ctx.rust_tools = true;
            let out = match action {
                ImplCmd::Map => ctx.call("rust_map", &json!({})),
                ImplCmd::Outline { file } => ctx.call("rust_outline", &json!({"path": file})),
                ImplCmd::Sites { kernel } => ctx.call("launch_sites", &json!({"kernel": kernel})),
                ImplCmd::Ops { op } => ctx.call("op_lowering", &json!({"op": op})),
            };
            println!("{out}");
        }
        Command::Ref { action, common } => {
            let ctx = setup(&cli, common, "cli")?;
            let out = match action {
                RefCmd::Kernels { kernel } => ctx.call("reference_kernels", &json!({"kernel": kernel})),
                RefCmd::Search { query, backend, limit } => ctx.call("reference_search", &json!({"query": query, "backend": backend, "limit": limit})),
            };
            println!("{out}");
        }
        Command::Scheme { code } => {
            let h = tensorlisp_evolve::scheme::handle()?;
            let out = h.run("cli", code, 30_000)?;
            if out.starts_with("error:") {
                bail!("{out}");
            }
            println!("{out}");
        }
    }
    Ok(())
}
