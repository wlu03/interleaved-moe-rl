"""Tests for the offline analysis module (src/analysis.py).

Plotting needs matplotlib (Modal-only), so these exercise the pure parsing /
series-extraction logic against synthetic records that mirror the Tracker's
real key schema.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import (
    RunRecords,
    drift_gap_summary,
    expert_load_at,
    layer_index,
    load_runs,
    per_layer_series,
)


def _records():
    # Two MoE layers (0, 2) and two dense (1, 3), a few steps of metrics.
    return [
        {"step": 0, "mean_reward": 0.1, "loss": 2.0,
         "grad/by_layer/L0": 1.0, "grad/by_layer/L1": 0.5,
         "routing/L0/entropy": 1.3, "routing/L0/load/E0": 0.3, "routing/L0/load/E1": 0.7},
        {"step": 0, "drift/cka/L0": 0.99, "drift/cka/L1": 0.98,
         "drift/procrustes/L0": 0.1,
         "drift/cka/by_kind/moe_mean": 0.99, "drift/cka/by_kind/dense_mean": 0.98},
        {"step": 5, "mean_reward": 0.4, "loss": 1.5,
         "grad/by_layer/L0": 0.8, "grad/by_layer/L1": 0.4,
         "routing/L0/entropy": 1.1, "routing/L0/load/E0": 0.4, "routing/L0/load/E1": 0.6},
        {"step": 5, "drift/cka/L0": 0.80, "drift/cka/L1": 0.95,
         "drift/cka/by_kind/moe_mean": 0.80, "drift/cka/by_kind/dense_mean": 0.95},
    ]


@pytest.fixture
def run():
    return RunRecords(name="moe_run", records=_records())


# RunRecords
class TestRunRecords:
    def test_from_file_uses_parent_dir_name(self, tmp_path):
        d = tmp_path / "dense_baseline"
        d.mkdir()
        (d / "records.json").write_text(json.dumps(_records()))
        run = RunRecords.from_file(d / "records.json")
        assert run.name == "dense_baseline"

    def test_series_sorted_and_filtered(self, run):
        steps, vals = run.series("mean_reward")
        assert steps == [0, 5]
        assert vals == [0.1, 0.4]

    def test_series_skips_missing_key(self, run):
        steps, _ = run.series("does_not_exist")
        assert steps == []

    def test_series_drops_nan(self):
        run = RunRecords("r", [
            {"step": 0, "drift/cka/L0": float("nan")},
            {"step": 1, "drift/cka/L0": 0.5},
        ])
        steps, vals = run.series("drift/cka/L0")
        assert steps == [1] and vals == [0.5]

    def test_keys_matching(self, run):
        keys = run.keys_matching(r"drift/cka/L\d+$")
        assert "drift/cka/L0" in keys and "drift/cka/L1" in keys
        assert "drift/cka/by_kind/moe_mean" not in keys

    def test_latest(self, run):
        assert run.latest("mean_reward") == 0.4
        assert run.latest("missing") is None


# layer helpers
class TestLayerHelpers:
    def test_layer_index(self):
        assert layer_index("drift/cka/L2") == 2
        assert layer_index("grad/by_layer/L0") == 0
        assert layer_index("loss") is None

    def test_per_layer_series(self, run):
        series = per_layer_series(run, "drift/cka")
        assert set(series) == {0, 1}
        steps, vals = series[0]
        assert steps == [0, 5] and vals == [0.99, 0.80]

    def test_per_layer_series_excludes_by_kind(self, run):
        # 'drift/cka/by_kind/moe_mean' must not be parsed as a layer
        series = per_layer_series(run, "drift/cka")
        assert all(isinstance(k, int) for k in series)


# expert load
class TestExpertLoad:
    def test_latest_expert_load(self, run):
        loads = expert_load_at(run, layer_idx=0)
        assert loads == {0: 0.4, 1: 0.6}

    def test_expert_load_at_specific_step(self, run):
        loads = expert_load_at(run, layer_idx=0, step=0)
        assert loads == {0: 0.3, 1: 0.7}

    def test_missing_layer_empty(self, run):
        assert expert_load_at(run, layer_idx=9) == {}


# drift gap
class TestDriftGap:
    def test_gap_dense_minus_moe(self, run):
        summary = drift_gap_summary(run)
        assert summary["moe_cka"] == 0.80
        assert summary["dense_cka"] == 0.95
        # MoE drifted more (lower CKA), so dense - moe is positive
        assert summary["dense_minus_moe"] == pytest.approx(0.15)

    def test_gap_none_when_missing(self):
        run = RunRecords("r", [{"step": 0, "loss": 1.0}])
        summary = drift_gap_summary(run)
        assert summary["dense_minus_moe"] is None


# load_runs
class TestLoadRuns:
    def test_load_from_directory(self, tmp_path):
        for name in ["moe_interleaved", "dense_baseline"]:
            d = tmp_path / name
            d.mkdir()
            (d / "records.json").write_text(json.dumps(_records()))
        runs = load_runs(tmp_path)
        assert {r.name for r in runs} == {"moe_interleaved", "dense_baseline"}

    def test_load_from_list(self, tmp_path):
        d = tmp_path / "run_a"
        d.mkdir()
        (d / "records.json").write_text(json.dumps(_records()))
        runs = load_runs([d / "records.json"])
        assert len(runs) == 1 and runs[0].name == "run_a"
