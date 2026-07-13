# Compute Efficiency Optimization (G2)
# Goal: Refactor firmware from tiled MVM to single-tile for dim=4
DIM=4
HIDDEN=4
NUM_HEAD=2
SEQ_LENS="2 6"
BASELINE_FILE="jimu-dse/baseline/bert_layer.c"
SKILLS="dim-optimize"
PRIMARY_METRIC="test_pass"
# Secondary metric: mv_mul_count (already optimal at baseline but agent may
# reduce firmware complexity / instruction count)
