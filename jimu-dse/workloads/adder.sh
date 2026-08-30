WORKLOAD_NAME="adder_140p"
MAKE_TARGET="adder_140p"
TARGET_FILE="adderboard/firmware/adder_140p.c"
TEST_VERIFY_CMD="python3 -m pytest adderboard/tests/test_phase2.py -k \"140p and first_step\" --no-header -q -rs"
TEST_CONVERGE_CMD="python3 -m pytest adderboard/tests/test_phase2.py -k \"140p and (first_step or autoregressive)\" --no-header -q -rs"
TEST_GATE_CMD="python3 -m pytest adderboard/tests/test_phase2.py -k \"140p and (first_step or autoregressive)\" --no-header -q -rs"
EXPECTED_GATE_TESTS_ALL=7
EXPECTED_GATE_TESTS_DIM4=7
