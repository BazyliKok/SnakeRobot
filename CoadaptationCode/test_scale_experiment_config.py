import os
import sys
import types
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if "gymnasium" not in sys.modules:
    gymnasium_stub = types.ModuleType("gymnasium")
    gymnasium_stub.Env = object
    gymnasium_stub.envs = types.SimpleNamespace(
        register=lambda *args, **kwargs: None,
    )
    gymnasium_stub.spaces = types.SimpleNamespace(
        Box=lambda *args, **kwargs: types.SimpleNamespace(
            low=kwargs.get("low"),
            high=kwargs.get("high"),
            shape=kwargs.get("shape"),
            dtype=kwargs.get("dtype"),
        )
    )
    sys.modules["gymnasium"] = gymnasium_stub
    sys.modules["gymnasium.spaces"] = gymnasium_stub.spaces

if "motorssynced" not in sys.modules:
    motorssynced_stub = types.ModuleType("motorssynced")
    motorssynced_stub.MotorsSynced = object
    sys.modules["motorssynced"] = motorssynced_stub

if "optitrack" not in sys.modules:
    optitrack_stub = types.ModuleType("optitrack")
    optitrack_stub.Optitrack = object
    sys.modules["optitrack"] = optitrack_stub

if "scipy" not in sys.modules:
    scipy_stub = types.ModuleType("scipy")
    scipy_interpolate_stub = types.ModuleType("scipy.interpolate")
    scipy_interpolate_stub.interp1d = lambda *args, **kwargs: None
    scipy_spatial_stub = types.ModuleType("scipy.spatial")
    scipy_transform_stub = types.ModuleType("scipy.spatial.transform")
    scipy_transform_stub.Rotation = object
    scipy_spatial_stub.transform = scipy_transform_stub
    scipy_stub.interpolate = scipy_interpolate_stub
    scipy_stub.spatial = scipy_spatial_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.interpolate"] = scipy_interpolate_stub
    sys.modules["scipy.spatial"] = scipy_spatial_stub
    sys.modules["scipy.spatial.transform"] = scipy_transform_stub

from snakeenv_thread_coadapt import SnakeEnv


def test_homogeneous_initial_designs_are_four_uniform_matched_extremes():
    designs = SnakeEnv.get_init_design_parameters("homogeneous")

    assert len(designs) == 4
    assert designs == [
        [0.63, 0.0, 0.63, 0.0],
        [0.63, 15.0, 0.63, 15.0],
        [0.90, 0.0, 0.90, 0.0],
        [0.90, 15.0, 0.90, 15.0],
    ]
    for design in designs:
        assert design[0] == design[2]
        assert design[1] == design[3]


def test_heterogeneous_initial_designs_are_four_alternating_matched_extremes():
    designs = SnakeEnv.get_init_design_parameters("heterogeneous")

    assert len(designs) == 4
    assert designs == [
        [0.63, 0.0, 0.90, 15.0],
        [0.90, 15.0, 0.63, 0.0],
        [0.63, 15.0, 0.90, 0.0],
        [0.90, 0.0, 0.63, 15.0],
    ]
    for design in designs:
        assert (design[0], design[1]) != (design[2], design[3])


def test_initial_designs_only_use_selected_widths_and_attack_angles():
    all_designs = (
        SnakeEnv.get_init_design_parameters("homogeneous")
        + SnakeEnv.get_init_design_parameters("heterogeneous")
    )

    widths = {design[index] for design in all_designs for index in (0, 2)}
    attack_angles = {design[index] for design in all_designs for index in (1, 3)}

    assert widths == {0.63, 0.90}
    assert attack_angles == {0.0, 15.0}


def test_attack_angle_bounds_and_feature_names_replace_tilt():
    assert SnakeEnv.scale_design_schema_version == 3
    assert SnakeEnv.design_parameter_names == [
        "A_width_ratio",
        "A_attack_angle_deg",
        "B_width_ratio",
        "B_attack_angle_deg",
    ]
    assert SnakeEnv.design_feature_names == [
        "A_width_norm",
        "A_attack_angle_norm",
        "B_width_norm",
        "B_attack_angle_norm",
        "Delta_width_norm",
        "Delta_attack_angle_norm",
    ]
    assert SnakeEnv.design_parameter_bounds == [
        (0.45, 0.90),
        (0.0, 15.0),
        (0.45, 0.90),
        (0.0, 15.0),
    ]


def test_optimization_bounds_and_homogeneous_expansion_clip_attack_angle():
    assert SnakeEnv.get_optimization_bounds("homogeneous") == [
        (0.45, 0.90),
        (0.0, 15.0),
    ]
    assert SnakeEnv.get_optimization_bounds("heterogeneous") == [
        (0.45, 0.90),
        (0.0, 15.0),
        (0.45, 0.90),
        (0.0, 15.0),
    ]

    assert SnakeEnv.expand_optimization_design([0.7, 20.0], "homogeneous") == [
        pytest.approx(0.7),
        15.0,
        pytest.approx(0.7),
        15.0,
    ]


def test_design_summary_and_module_expansion_use_attack_angle_columns():
    summary = SnakeEnv.design_summary([0.63, 5.0, 0.90, 15.0])

    assert summary["A_Attack_Angle_Deg"] == 5.0
    assert summary["B_Attack_Angle_Deg"] == 15.0
    assert summary["Attack_Angle_Delta"] == -10.0
    assert "A_Tilt_Deg" not in summary
    assert "B_Tilt_Deg" not in summary
    assert "Tilt_Delta" not in summary

    modules = SnakeEnv.expand_design_to_modules([0.63, 5.0, 0.90, 15.0])
    assert modules[0]["attack_angle_deg"] == 5.0
    assert modules[1]["attack_angle_deg"] == 15.0
    assert "tilt_deg" not in modules[0]


def test_default_active_terrains_are_carpet_and_foam(monkeypatch):
    monkeypatch.delenv("SNAKE_ACTIVE_TERRAINS", raising=False)
    train_source = (PROJECT_DIR / "train_coadapt.py").read_text()

    assert "os.getenv('SNAKE_ACTIVE_TERRAINS', 'carpet,foam')" in train_source


def test_pso_defaults_to_minimum_heterogeneity_for_heterogeneous_mode():
    pso_source = (PROJECT_DIR / "pso_batch.py").read_text()

    assert "'0.1' if str(design_mode).strip().lower() == 'heterogeneous' else '0.0'" in pso_source
    assert "SNAKE_MIN_HETEROGENEITY_DELTA" in pso_source
