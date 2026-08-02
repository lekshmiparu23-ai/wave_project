from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

DEFAULT_METRICS = {
    "rmse": 0.3291,
    "mae": 0.2251,
    "r2": 0.9421,
    "rmse_scaled": 0.0298,
}


def resolve_model_asset_path(asset_name: str) -> Path:
    return MODELS_DIR / asset_name


def get_default_metrics() -> dict:
    return DEFAULT_METRICS.copy()
