from pathlib import Path

from model_utils import MODELS_DIR, get_default_metrics, resolve_model_asset_path


def test_model_assets_directory_exists():
    assert MODELS_DIR.exists()


def test_model_asset_paths_resolve_under_models_dir():
    path = resolve_model_asset_path("wave_model.keras")
    assert path == MODELS_DIR / "wave_model.keras"
    assert path.parent == MODELS_DIR


def test_default_metrics_contains_expected_keys():
    metrics = get_default_metrics()
    assert set(metrics.keys()) == {"rmse", "mae", "r2", "rmse_scaled"}
    assert metrics["r2"] > 0.9
