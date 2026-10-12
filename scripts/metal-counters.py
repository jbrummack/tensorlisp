#!/usr/bin/env python3
"""Average Metal GPU counters over the busy samples of a Metal System Trace.

  xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters' \
      --output run.trace --time-limit 25s --launch -- ./bench-binary ...
  xcrun xctrace export --input run.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]' > run.xml
  python3 scripts/metal-counters.py a.xml b.xml        # one column per file

"Busy" is a sample with compute occupancy above 5%. Counter ids are Apple GPU (M1) ones; see the
gpu-counter-info table of the trace for the list on another chip.
"""
import sys, re, collections
from xml.etree.ElementTree import iterparse
names = {0:"ALU Limiter",1:"ALU Utilization",2:"F32 Utilization",3:"F16 Utilization",10:"Buffer Read Limiter",11:"Buffer Load Utilization",12:"Buffer Write Limiter",13:"Buffer Store Utilization",14:"Threadgroup Load Limiter",15:"Threadgroup Load Util",16:"Threadgroup Store Limiter",17:"Threadgroup Store Util",20:"Last Level Cache Limiter",21:"Last Level Cache Util",24:"Compute Occupancy",25:"GPU Read BW (GB/s)",26:"GPU Write BW (GB/s)",27:"MMU Limiter"}
def load(path):
    ids = {}
    by_ts = collections.defaultdict(dict)
    for ev, el in iterparse(path, events=("end",)):
        if el.tag != "row": continue
        vals = []
        for c in el:
            if c.get("ref") is not None:
                vals.append(ids[c.get("ref")])
            else:
                v = c.text
                if c.get("id") is not None: ids[c.get("id")] = v
                vals.append(v)
        ts, cid, val = vals[0], int(vals[1]), float(vals[2])
        by_ts[ts][cid] = val
        el.clear()
    return by_ts
res = {}
for k in sys.argv[1:]:
    d = load(f"{k}")
    busy = [c for c in d.values() if c.get(24, 0) > 5]
    n = len(busy)
    res[k] = (n, {cid: sum(c.get(cid, 0) for c in busy) / max(n, 1) for cid in names})
print(f"{'counter':28s}" + "".join(f"{k.split('/')[-1][:14]:>16s}" for k in res))
print(f"{'busy samples':28s}" + "".join(f"{res[k][0]:16d}" for k in res))
for cid, nm in names.items():
    print(f"{nm:28s}" + "".join(f"{res[k][1][cid]:16.1f}" for k in res))
