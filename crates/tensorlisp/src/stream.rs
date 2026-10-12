//! Streams between a running pipeline and its caller.
//!
//! A pipeline argument of kind `stream` is an input stream: the caller pushes [`Chunk`]s into the
//! [`StreamSender`] of [`input_stream`] and the pipeline takes them with `(stream-recv s)`. When no
//! chunk is there yet the pipeline *yields*: it leaves the Scheme thread, which stays free for
//! other work, and is resumed once something arrives (or the sender is dropped, when `stream-recv`
//! returns `#f`). Output goes the other way: `(emit 'name value)` calls the [`Emit`] given to
//! [`crate::Model::pipeline_streaming`].
use std::{
    collections::VecDeque,
    ffi::CString,
    sync::{Arc, Condvar, Mutex},
};

use autopro::audio::Audio;
use image::RgbImage;
use ndarray::ArrayD;

/// One item of a stream.
#[derive(Clone)]
pub enum Chunk {
    Array(ArrayD<f32>),
    Text(String),
    Audio(Audio),
    Image(RgbImage),
}

/// Receives what a pipeline emits: the output stream's name and the chunk. Returns false when
/// nobody listens any more, which aborts the pipeline.
pub type Emit = Arc<dyn Fn(&str, Chunk) -> bool + Send + Sync>;

struct Queue {
    chunks: VecDeque<Chunk>,
    closed: bool,
}

pub(crate) struct Inner {
    queue: Mutex<Queue>,
    ready: Condvar,
    on_take: Box<dyn Fn() + Send + Sync>,
    /// Host id of the chunk `poll` returned last (for the Scheme side to fetch).
    pub(crate) last: Mutex<i64>,
    /// The text of the last text chunk.
    pub(crate) text: Mutex<CString>,
}

pub(crate) enum Poll {
    Chunk(Chunk),
    /// Nothing yet; the sender is still there.
    Empty,
    /// Nothing more will come.
    Closed,
}

impl Inner {
    pub(crate) fn poll(&self) -> Poll {
        let mut q = self.queue.lock().unwrap();
        match q.chunks.pop_front() {
            Some(chunk) => {
                drop(q);
                (self.on_take)();
                Poll::Chunk(chunk)
            }
            None if q.closed => Poll::Closed,
            None => Poll::Empty,
        }
    }

    /// Blocks until `poll` would not return `Empty`.
    pub(crate) fn wait(&self) {
        let mut q = self.queue.lock().unwrap();
        while q.chunks.is_empty() && !q.closed {
            q = self.ready.wait(q).unwrap();
        }
    }
}

/// The receiving end of an input stream, passed to a pipeline as [`crate::RawInput::Stream`].
#[derive(Clone)]
pub struct StreamReader {
    pub(crate) inner: Arc<Inner>,
}

/// The sending end. Dropping it (or [`StreamSender::close`]) ends the stream.
pub struct StreamSender {
    inner: Arc<Inner>,
}

/// The stream was closed.
#[derive(Debug)]
pub struct Closed;

impl StreamSender {
    pub fn send(&self, chunk: Chunk) -> Result<(), Closed> {
        let mut q = self.inner.queue.lock().unwrap();
        if q.closed {
            return Err(Closed);
        }
        q.chunks.push_back(chunk);
        self.inner.ready.notify_all();
        Ok(())
    }

    pub fn close(&self) {
        self.inner.queue.lock().unwrap().closed = true;
        self.inner.ready.notify_all();
    }
}

impl Drop for StreamSender {
    fn drop(&mut self) {
        self.close();
    }
}

/// A channel for one input stream. `on_take` runs each time the pipeline takes a chunk, e.g. to
/// release a permit so a producer can send the next one (flow control).
pub fn input_stream(on_take: impl Fn() + Send + Sync + 'static) -> (StreamSender, StreamReader) {
    let inner = Arc::new(Inner {
        queue: Mutex::new(Queue { chunks: VecDeque::new(), closed: false }),
        ready: Condvar::new(),
        on_take: Box::new(on_take),
        last: Mutex::new(0),
        text: Mutex::new(CString::default()),
    });
    (StreamSender { inner: inner.clone() }, StreamReader { inner })
}

/// What a pipeline asks its caller to sample with `(client-sample logits history k full)`.
pub struct SampleRequest {
    /// How many samples this run has asked for before (0 for the first).
    pub step: u32,
    /// The `k` highest logits as (token id, logit), highest first.
    pub candidates: Vec<(i32, f32)>,
    /// Token ids generated so far, as the pipeline passed them.
    pub history: Vec<i32>,
    /// The whole logits array, if the pipeline asked for it.
    pub full: Option<ArrayD<f32>>,
}

/// Client-side logic a pipeline can call while it runs. Each call blocks the pipeline (not the
/// Scheme thread) until it returns.
pub trait Callbacks: Send + Sync {
    /// `(client-call name arg ...)`. Numbers arrive as 0-d arrays; the result may be an array,
    /// audio, an image or text, and `None` is `(void)`.
    fn call(&self, name: &str, args: Vec<Chunk>) -> Result<Option<Chunk>, String>;

    /// `(client-sample logits ...)`: the token to use.
    fn sample(&self, request: SampleRequest) -> Result<i32, String>;
}
