# Dim16 BERT Large Baseline

The large optimization baseline uses `NATIVE_DIM=16`, `hidden_size=16`,
`seq_len=16`, and one attention head. Its input tensor contains 256 elements,
which is 10.67 times the 24 elements in the historical dim4/seq6 workload.
The six projection/FFN matrices and associated vector parameters contain about
12.5 times as many elements as the small baseline.

## Layout and correctness

The goal selects `target.layout: packed-v2`. The shared layout calculator places
weights first, then LayerNorm parameters and non-overlapping Q/K/V, scratch,
residual, output, and unit-vector regions. Tile rows use a 16-element stride.
The existing goals omit `target.layout` and therefore retain `legacy-v1` and
their historical addresses.

The configuration is single-tile and single-head. One head is intentional: the
current vector-mask register has eight repeating mask bits and cannot select two
independent eight-element halves of a 16-element vector.

## Timing boundary

`jimu-rtl-dim16.yaml` retains the existing 8-byte/cycle external DRAM model and
uses 16 MVU/vector lanes with 32-byte/cycle on-chip transfers. Controller timing
comes from the existing HDL-derived formulas, but the dimension-16 profile has
not been calibrated against a dimension-16 functional RTL implementation or
FPGA. It is suitable for relative firmware comparisons, not absolute hardware
performance claims.

The only promotion metric is `rtl_predicted_npu_cycles`. DRAM elements,
instruction counts, utilization, overlap, and stall counters are diagnostics.

## Commands

```bash
python3 jimu-dse/scripts/closed_loop.py validate-config \
  --goal rtl-cycle-optimization-large
python3 jimu-dse/scripts/closed_loop.py render-prompt \
  --goal rtl-cycle-optimization-large
bash jimu-dse/scripts/npu_closed_loop.sh \
  --goal rtl-cycle-optimization-large --agent opencode
```

Start experimental runs with one iteration and inspect the baseline schedule
before increasing the loop count.
