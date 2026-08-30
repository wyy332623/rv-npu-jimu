---
name: dag-analyze
description: Read DAG outputs to identify DRAM save-load pairs for VRF caching
license: MIT
---

# DAG Analyze Skill

## Inputs

The pipeline produces these files at `_out/`:

| File | Content |
|------|---------|
| `micro_op_dag.txt` | All micro-ops with their defs/uses and edges |
| `dram_clusters.txt` | DRAM-flow clusters with FLOPs, bytes, AI |

## How to Read the DAG

### Identifying Save-Load Pairs

A save-load pair is a `DRAM_STORE` followed by a `DRAM_LOAD` of the **same DRAM address**, with no intervening write to that address.

In `micro_op_dag.txt`:
```
 33 DRAM_STORE       DRAM_STORE           [71-72]      uses=VRF[7][0] defs=DRAM[0x300]
    <- [30 VV_ADD] via VRF[7][0]
    <- [32 VREG_MOVE] via ('pipe',)
 34 DRAM_STORE       DRAM_STORE           [73-74]      uses=VRF[8][0] defs=DRAM[0x308]
...
150 DRAM_LOAD        DRAM_LOAD            [340-341]    uses=DRAM[0x300] defs=VRF[18][0]
    <- [33 DRAM_STORE] via DRAM[0x300]
```

The edge `<- [33 DRAM_STORE] via DRAM[0x300]` at node 150 shows that node 150 reads what node 33 wrote. This is a **DRAM save-load pair** — data goes to DRAM then comes right back.

### Identifying the Functions to Modify

1. Find the DRAM_STORE — check the event indices (`[71-72]`) and find the corresponding firmware code
2. Find the DRAM_LOAD — check the event index and find the corresponding firmware code
3. The DRAM address tells you which tensor in the firmware's scratch space the NPU is targeting. Look at the C macros defining DRAM mapping layout.

### Interpreting Cluster View

In `dram_clusters.txt`:
```
  LoadB  StoreB  FLOPs   AI
   40       8      48    1.0  K Proj: loads WEIGHT+X, saves K
   32       8      60    1.5  Attn Score: loads K+Q+V, saves prob
```

- `LoadB` is DRAM → NPU traffic
- `StoreB` is NPU → DRAM traffic
- `FLOPs` is computation performed
- `AI = FLOPs / (LoadB + StoreB)` — higher is better

Low AI clusters (< 1.0) are memory-bound and are the primary optimization targets.

## Output

Produce a list of eligible save-load pairs:
```
Save-load pairs:
  DRAM_STORE[0x300] at node 33 → DRAM_LOAD[0x300] at node 150 (K[0])
  DRAM_STORE[0x308] at node 34 → DRAM_LOAD[0x308] at node 177 (K[0] tr1)
  DRAM_STORE[0x400] at node 89 → DRAM_LOAD[0x400] at node 161 (V[0])
  ...
```
