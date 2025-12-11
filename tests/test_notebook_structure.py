import textwrap
from pathlib import Path

import nbformat
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "stonks-project-copy (12).ipynb"


def load_notebook():
    if not NOTEBOOK_PATH.exists():
        pytest.skip("Notebook is missing; nothing to validate.")
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def extract_config_cell(nb):
    for cell in nb.cells:
        if cell.cell_type == "code" and "class Config" in cell.source:
            return cell.source
    raise AssertionError("Could not find Config definition in notebook.")


def build_cfg():
    nb = load_notebook()
    source = extract_config_cell(nb)
    ns: dict = {}
    prelude = "from dataclasses import dataclass\nfrom typing import List, Dict\nimport pandas as pd\n"
    exec(textwrap.dedent(prelude + source), ns)
    return ns["CFG"]


def test_notebook_has_required_sections():
    nb = load_notebook()
    headers = [cell.source.strip() for cell in nb.cells if cell.cell_type == "markdown"]
    expected = {
        "## 1) Configuration",
        "## 2) Data Ingestion",
        "## 3) Feature Engineering (per ticker)",
        "## 4) Labels, Panel, Cross-Sectional Targets & Ranks",
        "## 5) Benchmark State / Regime Filter",
        "## 6) Metrics + Helpers",
        "## 7) Walk-Forward Backtest with Hold-Threshold",
        "## 8) Run Backtest",
        "## 9) Benchmark & Plots",
        "## 10) Metrics & Artifact Export",
    }
    for title in expected:
        assert any(h.startswith(title) for h in headers), f"Missing section {title}"


def test_config_defaults_are_reasonable():
    cfg = build_cfg()
    assert len(cfg.universe) >= 40
    assert cfg.benchmark.upper() == "SPY"
    assert cfg.horizon_days > 0
    assert 0 < cfg.cs_target_quantile < 1
    assert cfg.top_k >= 1
    assert cfg.hold_threshold >= 0.0
    assert cfg.xgb_params["n_estimators"] >= 100
