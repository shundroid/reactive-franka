import json
from types import SimpleNamespace

import pytest

from reactive_franka.main import run
from reactive_franka.profiling import Profiler


def test_sync_excludes_prior_work_and_includes_completion(monkeypatch):
    clock = [0]
    waits = []

    def synchronize():
        waits.append(clock[0])
        clock[0] += 5_000_000

    monkeypatch.setattr("reactive_franka.profiling.synchronize_cuda", synchronize)
    monkeypatch.setattr("reactive_franka.profiling.time.perf_counter_ns", lambda: clock[0])
    profiler = Profiler("sync")
    with profiler.span("arbitrary_gpu_stage", gpu=True):
        clock[0] += 3_000_000
    assert len(waits) == 2
    assert profiler.samples["arbitrary_gpu_stage"] == [8]


@pytest.mark.parametrize("mode", [None, "wall"])
def test_wall_and_disabled_do_not_synchronize(tmp_path, monkeypatch, mode):
    def unexpected():
        raise AssertionError("unexpected CUDA synchronization")

    monkeypatch.setattr("reactive_franka.profiling.synchronize_cuda", unexpected)
    profiler = Profiler(mode)
    with profiler.span("arbitrary_gpu_stage", gpu=True):
        pass
    profiler.save(tmp_path / "profile.json")
    assert (tmp_path / "profile.json").exists() == (mode is not None)
    assert bool(profiler.samples) == (mode is not None)


def test_names_are_opaque_and_output_is_only_json(tmp_path):
    profiler = Profiler("wall")
    for stage in ("kalman.total@prediction", "kalman.total@measurement", "custom@anything"):
        with profiler.span(stage):
            pass
    profiler.save(tmp_path / "profile.json")
    assert [p.name for p in tmp_path.iterdir()] == ["profile.json"]
    report = json.loads((tmp_path / "profile.json").read_text())
    assert set(report) == {"mode", "unit", "stages"}
    assert set(report["stages"]) == set(profiler.samples)
    for stage, result in report["stages"].items():
        assert result["count"] == 1
        assert result["samples_ms"] == profiler.samples[stage]
        assert result["median_ms"] == result["samples_ms"][0]


def test_exception_records_elapsed_time_and_propagates():
    profiler = Profiler("wall")
    with pytest.raises(RuntimeError, match="failure"), profiler.span("failing_stage"):
        raise RuntimeError("failure")
    assert len(profiler.samples["failing_stage"]) == 1


def cpu_args(output):
    return SimpleNamespace(
        headless=True,
        gui_light=False,
        controller="bullet-ik",
        perception="sim-mask",
        weights="",
        confidence=0.25,
        steps=12,
        cycles=0,
        fast=True,
        output=output,
        profile="wall",
        profile_warmup_steps=3,
    )


def test_initialization_failure_preserves_error_and_saves_profile(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("simulation initialization failed")

    monkeypatch.setattr("reactive_franka.main.Simulation", fail)
    args = cpu_args(tmp_path / "new-output")
    with pytest.raises(RuntimeError, match="simulation initialization failed"):
        run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["error"] == "simulation initialization failed"
    assert summary["steps"] == 0
    report = json.loads((args.output / "profile.json").read_text())
    assert report["stages"] == {}


def test_main_warmup_and_combined_sim_step_preserve_behavior(tmp_path):
    args = cpu_args(tmp_path / "profiled")
    profiled = run(args)
    report = json.loads((args.output / "profile.json").read_text())
    stages = report["stages"]
    assert stages["sim_step.total"]["count"] == 9
    assert stages["camera.total"]["count"] == 3
    assert stages["perception.total"]["count"] == 3
    assert stages["step.total@perception"]["count"] == 3
    assert stages["step.total@control"]["count"] == 6
    assert "collision.total" not in stages and "physics.total" not in stages
    assert "yolo.inference" not in stages
    assert sorted(p.name for p in args.output.glob("profile*")) == ["profile.json"]
    args.output = tmp_path / "unprofiled"
    args.profile = None
    unprofiled = run(args)
    assert not (args.output / "profile.json").exists()
    profiled.pop("wall_seconds")
    unprofiled.pop("wall_seconds")
    assert profiled == unprofiled
