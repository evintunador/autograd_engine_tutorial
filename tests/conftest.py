def pytest_addoption(parser):
    parser.addoption(
        "--seed", action="store", type=int, default=0,
        help="Base RNG seed for generating test inputs (default: 0).",
    )
    parser.addoption(
        "--heatmaps", action="store_true", default=False,
        help="On a comparison failure, write diff heatmap PNGs to tests/heatmaps/.",
    )
