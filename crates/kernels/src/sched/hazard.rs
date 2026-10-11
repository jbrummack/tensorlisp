//! Hazard analysis: cuts a launch sequence into groups separated by barriers.
//!
//! Launches in one group have no data dependency on each other (any order, or
//! concurrently); a barrier goes between groups. This is the backend-neutral
//! half of what ggml-metal does with `ggml_mem_ranges`.

/// Where a byte range lives. See docs/design/scheduler.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// Read-only weights and constants shared by every session.
    Model(u32),
    /// State owned by one session (KV caches, ...).
    Session(u32),
    /// The submission's arena: inputs, outputs, activations, scratch.
    Step,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub space: Space,
    pub off: u64,
    pub len: u64,
}

impl Range {
    pub fn new(space: Space, off: u64, len: u64) -> Self {
        Range { space, off, len }
    }

    pub fn overlaps(&self, other: &Range) -> bool {
        self.space == other.space
            && self.len > 0
            && other.len > 0
            && self.off < other.off + other.len
            && other.off < self.off + self.len
    }
}

/// What one launch reads and writes.
#[derive(Clone, Debug, Default)]
pub struct Footprint {
    pub reads: Vec<Range>,
    pub writes: Vec<Range>,
}

impl Footprint {
    fn conflicts_with(&self, group: &Footprint) -> bool {
        let any = |a: &[Range], b: &[Range]| a.iter().any(|x| b.iter().any(|y| x.overlaps(y)));
        any(&self.reads, &group.writes) // RAW
            || any(&self.writes, &group.reads) // WAR
            || any(&self.writes, &group.writes) // WAW
    }
}

/// Upper bound on launches per group, to keep the pairwise check cheap.
pub const MAX_GROUP: usize = 128;

/// Groups launch indices in program order: a launch joins the current group
/// unless it conflicts with it (or the group is full), which starts a new one.
pub fn group(footprints: &[Footprint]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut acc = Footprint::default();
    let mut current: Vec<usize> = Vec::new();
    for (i, f) in footprints.iter().enumerate() {
        if !current.is_empty() && (current.len() >= MAX_GROUP || f.conflicts_with(&acc)) {
            groups.push(std::mem::take(&mut current));
            acc = Footprint::default();
        }
        current.push(i);
        acc.reads.extend_from_slice(&f.reads);
        acc.writes.extend_from_slice(&f.writes);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Groups by dependency depth: a launch goes to the group after the latest group holding a launch it
/// conflicts with (group 0 when it conflicts with none), so independent branches of a graph (the
/// q, k and v paths of attention) run together whatever order the program listed them in. Within
/// a group launches keep program order; between groups there is a barrier, so any order of
/// groups that respects every conflict gives the results of program order.
pub fn level_group(footprints: &[Footprint]) -> Vec<Vec<usize>> {
    let mut level = vec![0usize; footprints.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..footprints.len() {
        let mut l = 0;
        for j in (0..i).rev() {
            // earlier launches in lower groups cannot raise the level past what a closer one set
            if level[j] + 1 > l && footprints[i].conflicts_with(&footprints[j]) {
                l = level[j] + 1;
            }
        }
        // a full group spills into the next one (their members stay independent of each other)
        while groups.get(l).is_some_and(|g| g.len() >= MAX_GROUP) {
            l += 1;
        }
        level[i] = l;
        if groups.len() <= l {
            groups.resize(l + 1, Vec::new());
        }
        groups[l].push(i);
    }
    groups.retain(|g| !g.is_empty());
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(off: u64, len: u64) -> Range {
        Range::new(Space::Step, off, len)
    }

    fn fp(reads: &[Range], writes: &[Range]) -> Footprint {
        Footprint { reads: reads.to_vec(), writes: writes.to_vec() }
    }

    #[test]
    fn independent_branches_share_a_group() {
        let w = Range::new(Space::Model(0), 0, 100);
        // k = x*w, v = x*w: both read the same input and weights, write apart.
        let groups = group(&[fp(&[r(0, 64), w], &[r(100, 64)]), fp(&[r(0, 64), w], &[r(200, 64)])]);
        assert_eq!(groups, vec![vec![0, 1]]);
    }

    #[test]
    fn dependencies_cut_groups() {
        // RAW, WAR, WAW each need a barrier.
        let raw = group(&[fp(&[], &[r(0, 8)]), fp(&[r(0, 8)], &[r(8, 8)])]);
        assert_eq!(raw, vec![vec![0], vec![1]]);
        let war = group(&[fp(&[r(0, 8)], &[r(8, 8)]), fp(&[], &[r(0, 8)])]);
        assert_eq!(war, vec![vec![0], vec![1]]);
        let waw = group(&[fp(&[], &[r(0, 8)]), fp(&[], &[r(4, 8)])]);
        assert_eq!(waw, vec![vec![0], vec![1]]);
    }

    #[test]
    fn spaces_and_empty_ranges_do_not_conflict() {
        let a = Range::new(Space::Session(0), 0, 8);
        let b = Range::new(Space::Session(1), 0, 8);
        assert_eq!(group(&[fp(&[], &[a]), fp(&[], &[b])]), vec![vec![0, 1]]);
        assert_eq!(group(&[fp(&[], &[r(0, 0)]), fp(&[r(0, 8)], &[])]), vec![vec![0, 1]]);
    }

    #[test]
    fn a_chain_is_one_group_per_link() {
        let fps: Vec<Footprint> = (0..5).map(|i| fp(&[r(i * 8, 8)], &[r((i + 1) * 8, 8)])).collect();
        assert_eq!(group(&fps).len(), 5);
    }

    #[test]
    fn levels_run_independent_branches_together() {
        // a -> b -> c (a chain) and x -> y, listed one branch after the other
        let fps = [
            fp(&[], &[r(0, 8)]),            // a
            fp(&[r(0, 8)], &[r(8, 8)]),     // b
            fp(&[r(8, 8)], &[r(16, 8)]),    // c
            fp(&[], &[r(100, 8)]),          // x
            fp(&[r(100, 8)], &[r(108, 8)]), // y
        ];
        assert_eq!(level_group(&fps), vec![vec![0, 3], vec![1, 4], vec![2]]);
        // the greedy grouping only looks at the launches next to each other
        assert_eq!(group(&fps).len(), 4);
    }

    #[test]
    fn levels_respect_every_conflict() {
        let mut seed = 5;
        for _ in 0..300 {
            let n = 1 + lcg(&mut seed) % 60;
            let fps: Vec<Footprint> = (0..n)
                .map(|_| {
                    let range = |s: &mut u64| r(lcg(s) % 64 * 4, 4 * (1 + lcg(s) % 6));
                    let nr = lcg(&mut seed) % 3;
                    let nw = 1 + lcg(&mut seed) % 2;
                    Footprint { reads: (0..nr).map(|_| range(&mut seed)).collect(), writes: (0..nw).map(|_| range(&mut seed)).collect() }
                })
                .collect();
            let groups = level_group(&fps);
            let mut at = vec![usize::MAX; fps.len()];
            for (gi, g) in groups.iter().enumerate() {
                for &i in g {
                    at[i] = gi;
                }
            }
            assert!(at.iter().all(|&g| g != usize::MAX), "every launch is placed");
            for b in 0..fps.len() {
                for a in 0..b {
                    if fps[b].conflicts_with(&fps[a]) {
                        assert!(at[a] < at[b], "launch {a} must run before {b}");
                    }
                }
            }
        }
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// Reordering inside groups must not change results: any two launches in
    /// the same group are independent, and groups preserve program order.
    #[test]
    fn random_footprints_are_independent_within_a_group() {
        let mut seed = 11;
        for _ in 0..300 {
            let n = 1 + lcg(&mut seed) % 60;
            let fps: Vec<Footprint> = (0..n)
                .map(|_| {
                    let range = |s: &mut u64| r(lcg(s) % 64 * 4, 4 * (1 + lcg(s) % 6));
                    let nr = lcg(&mut seed) % 3;
                    let nw = 1 + lcg(&mut seed) % 2;
                    Footprint {
                        reads: (0..nr).map(|_| range(&mut seed)).collect(),
                        writes: (0..nw).map(|_| range(&mut seed)).collect(),
                    }
                })
                .collect();
            let groups = group(&fps);
            let flat: Vec<usize> = groups.iter().flatten().copied().collect();
            assert_eq!(flat, (0..fps.len()).collect::<Vec<_>>(), "program order is preserved");
            for g in &groups {
                for (ai, &a) in g.iter().enumerate() {
                    for &b in &g[ai + 1..] {
                        assert!(!fps[b].conflicts_with(&fps[a]), "launches {a} and {b} conflict inside a group");
                    }
                }
            }
        }
    }
}
