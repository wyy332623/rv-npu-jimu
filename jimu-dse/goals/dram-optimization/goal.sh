# DRAM Traffic Optimization (G1)
# Goal: Reduce DRAM bytes via VRF cache

HIDDEN=4
NUM_HEAD=2
SKILLS="vrf-cache"
PRIMARY_METRIC="total_bytes"

if [[ "${WORKLOAD}" == "adder" ]]; then
    DIM=4
    HIDDEN=4
    NUM_HEAD=1
    # Prompt-only and first decode step: two distinct concrete DAGs are
    # required for measured cross-length reuse evidence.
    SEQ_LENS="24 25"
    BASELINE_FILE="jimu-dse/baseline/adder_140p.c"
else
    DIM=2
    SEQ_LENS="2 6"
    BASELINE_FILE="jimu-dse/baseline/bert_layer.c"
fi
