# Compute Efficiency Optimization (G2)
# Goal: Refactor firmware from tiled MVM to single-tile

DIM=4
HIDDEN=4
NUM_HEAD=2
SKILLS="dim-optimize"
PRIMARY_METRIC="test_pass"

if [[ "${WORKLOAD}" == "adder" ]]; then
    NUM_HEAD=1
    SEQ_LENS="24 25"
    BASELINE_FILE="jimu-dse/baseline/adder_140p.c"
else
    SEQ_LENS="2 6"
    BASELINE_FILE="jimu-dse/baseline/bert_layer.c"
fi
