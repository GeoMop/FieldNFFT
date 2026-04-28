"""
tests/conftest.py – Shared pytest configuration and fixtures.

Usage
-----
    pytest tests/                      # numerical assertions only (fast)
    pytest tests/ --plots              # also save figures to tests/plots/
    pytest tests/ --plots -v           # verbose with figures
"""

import pytest
from pathlib import Path


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--plots",
        action="store_true",
        default=False,
        help="Save comparison figures to tests/plots/ (for report).",
    )


@pytest.fixture
def plot_dir(request: pytest.FixtureRequest, tmp_path) -> Path | None:
    """Return the output directory for figures, or None when --plots is not set.

    When --plots is active the directory tests/plots/ is created if it does not
    exist.  Tests that receive this fixture should guard all plotting code with::

        if plot_dir is not None:
            ...
    """
    if request.config.getoption("--plots"):
        out = Path(__file__).parent / "plots"
        out.mkdir(exist_ok=True)
        return out
    return None