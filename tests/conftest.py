def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: slow test requiring full simulation")

def pytest_addoption(parser):
    """Register --instrument flag for op-by-op debug diagnostics."""
    parser.addoption(
        "--instrument", action="store_true", default=False,
        help="Enable NpuInstrumentor and print per-operator diagnostics"\
    )
