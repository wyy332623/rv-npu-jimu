# DRAM Traffic Optimization (G1)
# Goal: Reduce DRAM bytes at fixed dim=2 via VRF cache
DIM=2
HIDDEN=4
NUM_HEAD=2
SEQ_LENS="2 6"
BASELINE_FILE="jimu-dse/baseline/bert_layer.c"
SKILLS="vrf-cache"
PRIMARY_METRIC="total_bytes"
