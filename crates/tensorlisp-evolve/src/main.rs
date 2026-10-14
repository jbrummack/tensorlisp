//! `tl-evolve`: run agents that search for Tree optimization passes, evaluate candidates, and use the forum.
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use tensorlisp_evolve::agent::{self, AgentConfig};
use tensorlisp_evolve::embed::{CodeIndex, Embedder, HashEmbedder, HttpEmbedder};
use tensorlisp_evolve::eval::{self, EvalConfig, GateCommand};
use tensorlisp_evolve::forum::Forum;
use tensorlisp_evolve::kv::FsKv;
use tensorlisp_evolve::llm::Llm;
use tensorlisp_evolve::search::SearchIndex;
use tensorlisp_evolve::tools::ToolCtx;

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
        #[arg(long, default_value_t = 300_000)]
        context_chars: usize,
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
            let base = std::env::var("TL_EVOLVE_EMBED_BASE_URL").or_else(|_| std::env::var("TL_EVOLVE_BASE_URL")).or_else(|_| std::env::var("OPENAI_BASE_URL")).unwrap_or_else(|_| "https://openrouter.ai/api/v1".into());
            let key = ["TL_EVOLVE_EMBED_API_KEY", "TL_EVOLVE_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"].iter().find_map(|k| std::env::var(k).ok());
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
    match &cli.command {
        Command::Run { model, common, task, turns, context_chars, temperature, quiet } => {
            let default_author = format!("agent-{}", model.rsplit('/').next().unwrap_or(model));
            let ctx = setup(&cli, common, &default_author)?;
            let mut llm = Llm::from_env(Some(model))?;
            llm.temperature = *temperature;
            let workspace = ctx.workspace.strip_prefix(&ctx.repo).unwrap_or(&ctx.workspace).display().to_string();
            let task = task.clone().unwrap_or_else(|| format!("Find a pass or pipeline that makes the Tree kernels smaller or faster than the baseline (fold licm cse dce) on {}. Start from the forum.", common.target));
            let agent_cfg = AgentConfig {
                max_turns: *turns,
                system: agent::system_prompt(&common.target, &common.device, &ctx.author, &workspace),
                task,
                transcript: Some(ctx.workspace.join("transcript.jsonl")),
                max_context_chars: *context_chars,
                verbose: !*quiet,
            };
            eprintln!("{} on {} via {}; workspace {workspace}", ctx.author, model, llm.base_url);
            let r = agent::run(&llm, &ctx, &agent_cfg)?;
            eprintln!("{} turns, {} tool calls, {} prompt / {} completion tokens", r.turns, r.tool_calls, r.prompt_tokens, r.completion_tokens);
            println!("{}", r.summary.unwrap_or_else(|| "(the agent did not finish)".into()));
        }
        Command::Eval { file, pipeline, common, json } => {
            let ctx = setup(&cli, common, "cli")?;
            let code = std::fs::read_to_string(file).with_context(|| file.display().to_string())?;
            let pipeline: Vec<String> = pipeline.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect();
            let ev = eval::evaluate(&ctx.forum, &ctx.eval_cfg, &code, &pipeline)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&json!({"id": ev.id, "verdict": ev.verdict, "report": ev.report}))?);
            } else {
                print!("{}", eval::render(&ev));
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
