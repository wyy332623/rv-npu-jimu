# Combined Optimization (G3)
# Goal: Compute efficiency + DRAM traffic at dim=4, hidden=4
DIM=4
HIDDEN=4
NUM_HEAD=2
SEQ_LENS="2 6"
BASELINE_FILE="jimu-dse/baseline/bert_layer.c"
SKILLS="dim-optimize vrf-cache"
PRIMARY_METRIC="test_pass"
