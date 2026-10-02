//! Memory planning: assigns every tensor a buffer and offset.
//!
//! The arena holds the outputs of nodes and the scratch space a lowering
//! asks for. A tensor's block is reused once its last reader has run (strictly
//! after: a node reads its sources while writing its destination).

use crate::graph::{Graph, Storage, TensorId};

pub const ALIGN: u64 = 256;

pub fn align(n: u64) -> u64 {
    n.div_ceil(ALIGN) * ALIGN
}

/// A buffer of the executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufId {
    Weights(usize),
    State,
    Io,
    Arena,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Place {
    pub buf: BufId,
    pub off: u64,
}

pub struct Layout {
    /// Per tensor (views included).
    pub place: Vec<Place>,
    pub arena_size: u64,
    pub io_size: u64,
    /// Per graph input, in `Storage::Input` order: offset and bytes in the io buffer.
    pub inputs: Vec<(u64, u64)>,
    /// Per node position in `graph.nodes`: scratch offset in the arena, if it asked for some.
    pub scratch: Vec<Option<u64>>,
}

/// Where leaf data lives outside the arena.
pub struct Leaves<'a> {
    /// Weight index -> (buffer index, byte offset).
    pub weights: &'a [(usize, u64)],
    /// State index -> byte offset in the state buffer.
    pub states: &'a [u64],
}

struct Block {
    id: TensorId,
    first: usize,
    last: usize,
    size: u64,
}

pub fn plan(graph: &Graph, leaves: &Leaves, scratch_bytes: &dyn Fn(TensorId) -> u64) -> Layout {
    let n = graph.nodes.len();
    let nt = graph.tensors.len();

    // Liveness of arena roots, in node positions.
    let mut first = vec![usize::MAX; nt];
    let mut last = vec![0usize; nt];
    let touch = |first: &mut [usize], last: &mut [usize], root: TensorId, pos: usize| {
        first[root] = first[root].min(pos);
        last[root] = last[root].max(pos);
    };
    for (pos, &id) in graph.nodes.iter().enumerate() {
        let t = &graph.tensors[id];
        let (dst_root, _) = graph.root(id);
        touch(&mut first, &mut last, dst_root, pos);
        for s in t.src.iter().flatten() {
            let (r, _) = graph.root(*s);
            touch(&mut first, &mut last, r, pos);
        }
    }
    // A view op is a no-op that may be consumed far later: its readers already
    // touch the root. Keep results alive to the end.
    for &id in &graph.keep {
        let (r, _) = graph.root(id);
        last[r] = n;
    }

    let blocks: Vec<Block> = (0..nt)
        .filter(|&id| graph.tensors[id].view_src.is_none() && graph.tensors[id].storage == Storage::Temp && first[id] != usize::MAX)
        .map(|id| Block { id, first: first[id], last: last[id], size: align(graph.tensors[id].nbytes().max(1)) })
        .collect();
    // Scratch is a short-lived block of its own, owned by the node.
    let mut scratch_blocks = Vec::new();
    for (pos, &id) in graph.nodes.iter().enumerate() {
        let bytes = scratch_bytes(id);
        if bytes > 0 {
            scratch_blocks.push((pos, Block { id, first: pos, last: pos, size: align(bytes) }));
        }
    }

    // First-fit by start time over the lowest free offset.
    struct Live {
        off: u64,
        size: u64,
        last: usize,
    }
    let mut live: Vec<Live> = Vec::new();
    let mut arena = 0u64;
    let mut offsets = vec![0u64; nt];
    let mut scratch = vec![None; n];

    let mut order: Vec<(usize, bool, usize)> = Vec::new(); // (first, is_scratch, index)
    order.extend(blocks.iter().enumerate().map(|(i, b)| (b.first, false, i)));
    order.extend(scratch_blocks.iter().enumerate().map(|(i, (_, b))| (b.first, true, i)));
    order.sort();

    for (_, is_scratch, i) in order {
        let (first_pos, last_pos, size) = if is_scratch {
            let b = &scratch_blocks[i].1;
            (b.first, b.last, b.size)
        } else {
            let b = &blocks[i];
            (b.first, b.last, b.size)
        };
        live.retain(|l| l.last >= first_pos);
        live.sort_by_key(|l| l.off);
        let mut off = 0u64;
        for l in &live {
            if l.off >= off + size {
                break;
            }
            off = off.max(l.off + l.size);
        }
        live.push(Live { off, size, last: last_pos });
        arena = arena.max(off + size);
        if is_scratch {
            scratch[scratch_blocks[i].0] = Some(off);
        } else {
            offsets[blocks[i].id] = off;
        }
    }

    // Inputs live in the io buffer.
    let mut inputs = Vec::new();
    let mut io = 0u64;
    for t in &graph.tensors {
        if let (None, Storage::Input(i)) = (t.view_src, t.storage) {
            if inputs.len() <= i {
                inputs.resize(i + 1, (0, 0));
            }
            let bytes = t.nbytes();
            inputs[i] = (io, bytes);
            io += align(bytes.max(1));
        }
    }

    let place = (0..nt)
        .map(|id| {
            let (root, extra) = graph.root(id);
            let t = &graph.tensors[root];
            let base = match t.storage {
                Storage::Weight(w) => Place { buf: BufId::Weights(leaves.weights[w].0), off: leaves.weights[w].1 },
                Storage::State(s) => Place { buf: BufId::State, off: leaves.states[s] },
                Storage::Input(i) => Place { buf: BufId::Io, off: inputs[i].0 },
                Storage::Temp => Place { buf: BufId::Arena, off: offsets[root] },
            };
            Place { buf: base.buf, off: base.off + extra }
        })
        .collect();

    Layout { place, arena_size: arena, io_size: io, inputs, scratch }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Tensor, Ty};

    fn tensor(op: &str, ne0: i64, src: Vec<Option<TensorId>>, view: Option<(TensorId, u64)>, storage: Storage) -> Tensor {
        Tensor {
            ty: Ty::F32,
            ne: [ne0, 1, 1, 1],
            nb: [4, 4 * ne0 as u64, 4 * ne0 as u64, 4 * ne0 as u64],
            op: op.into(),
            op_params: [0; 16],
            src,
            view_src: view.map(|v| v.0),
            view_offs: view.map_or(0, |v| v.1),
            storage,
            name: String::new(),
        }
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    fn leaves() -> Leaves<'static> {
        Leaves { weights: &[], states: &[] }
    }

    #[test]
    fn a_chain_reuses_two_blocks() {
        let mut g = Graph::default();
        g.tensors.push(tensor("NONE", 256, vec![], None, Storage::Input(0)));
        for i in 0..6 {
            g.tensors.push(tensor("ADD", 256, vec![Some(i), Some(0)], None, Storage::Temp));
            g.nodes.push(i + 1);
        }
        g.keep.push(6);
        let layout = plan(&g, &leaves(), &|_| 0);
        // Each node reads its predecessor while writing: two blocks suffice, the input is separate.
        assert_eq!(layout.arena_size, 2 * align(1024));
        assert_eq!(layout.io_size, align(1024));
        assert_eq!(layout.inputs, vec![(0, 1024)]);
    }

    #[test]
    fn views_extend_the_life_of_their_source() {
        let mut g = Graph::default();
        g.tensors.push(tensor("NONE", 256, vec![], None, Storage::Input(0)));
        g.tensors.push(tensor("ADD", 256, vec![Some(0), Some(0)], None, Storage::Temp)); // 1: a
        g.tensors.push(tensor("VIEW", 128, vec![Some(1)], Some((1, 512)), Storage::Temp)); // 2: view of a's tail
        g.tensors.push(tensor("ADD", 256, vec![Some(0), Some(0)], None, Storage::Temp)); // 3: b
        g.tensors.push(tensor("ADD", 256, vec![Some(3), Some(3)], None, Storage::Temp)); // 4: c
        g.tensors.push(tensor("ADD", 128, vec![Some(2), Some(2)], None, Storage::Temp)); // 5: reads the view late
        g.nodes = vec![1, 2, 3, 4, 5];
        g.keep.push(5);
        let layout = plan(&g, &leaves(), &|_| 0);
        let (a, b, c) = (layout.place[1], layout.place[3], layout.place[4]);
        assert_eq!(layout.place[2].off, a.off + 512, "a view sits inside its source");
        let overlap = |x: Place, xs: u64, y: Place, ys: u64| x.off < y.off + ys && y.off < x.off + xs;
        // a is alive until node 5 reads its view, so b and c may not take its memory.
        assert!(!overlap(a, 1024, b, 1024) && !overlap(a, 1024, c, 1024));
    }

    #[test]
    fn random_graphs_never_overlap_live_tensors() {
        let mut seed = 7;
        for _ in 0..200 {
            let mut g = Graph::default();
            g.tensors.push(tensor("NONE", 64, vec![], None, Storage::Input(0)));
            let n = 5 + lcg(&mut seed) % 40;
            for i in 0..n as usize {
                let id = g.tensors.len();
                let a = (lcg(&mut seed) as usize) % (id);
                let b = (lcg(&mut seed) as usize) % (id);
                let size = 16 * (1 + lcg(&mut seed) as i64 % 40);
                g.tensors.push(tensor("ADD", size, vec![Some(a), Some(b)], None, Storage::Temp));
                g.nodes.push(id);
                let _ = i;
            }
            g.keep.push(g.tensors.len() - 1);
            let layout = plan(&g, &leaves(), &|id| if id % 7 == 0 { 100 } else { 0 });

            // Independent liveness: first write .. last read, by node position.
            let pos = |t: TensorId| g.nodes.iter().position(|&n| n == t);
            let mut live: Vec<(usize, usize, u64, u64)> = Vec::new(); // first, last, off, size
            for (p, &id) in g.nodes.iter().enumerate() {
                let last = g
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| g.tensors[**m].src.iter().flatten().any(|&s| s == id))
                    .map(|(q, _)| q)
                    .max()
                    .unwrap_or(p);
                let last = if g.keep.contains(&id) { g.nodes.len() } else { last };
                live.push((p, last, layout.place[id].off, g.tensors[id].nbytes()));
            }
            for (i, x) in live.iter().enumerate() {
                for y in &live[i + 1..] {
                    let time = x.0 <= y.1 && y.0 <= x.1;
                    let space = x.2 < y.2 + y.3 && y.2 < x.2 + x.3;
                    assert!(!(time && space), "live tensors overlap: {x:?} {y:?}");
                }
            }
            // A node never writes where it reads.
            for (p, &id) in g.nodes.iter().enumerate() {
                for s in g.tensors[id].src.iter().flatten() {
                    if *s != 0 && pos(*s).is_some() {
                        let (a, b) = (layout.place[id], layout.place[*s]);
                        let (sa, sb) = (g.tensors[id].nbytes(), g.tensors[*s].nbytes());
                        assert!(!(a.off < b.off + sb && b.off < a.off + sa), "node {p} writes over its source");
                    }
                }
            }
            for (p, scratch) in layout.scratch.iter().enumerate() {
                if let Some(off) = scratch {
                    let id = g.nodes[p];
                    assert_eq!(id % 7, 0);
                    assert_eq!(off % ALIGN, 0);
                }
            }
        }
    }
}
