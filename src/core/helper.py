from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyYAML is required for model config loading. Install with `pip install pyyaml`."
    ) from exc


_DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def _load_models_yaml(config_path: str | Path | None = None) -> List[dict]:
    cfg_path = Path(config_path) if config_path else _DEFAULT_MODEL_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"Invalid model config format in {cfg_path}: 'models' must be a list")
    return models


def load_model_registry(config_path: str | Path | None = None) -> Dict[str, dict]:
    registry: Dict[str, dict] = {}
    for item in _load_models_yaml(config_path):
        model_id = item.get("id")
        if not model_id:
            continue
        registry[model_id] = item
    return registry


def load_active_models(config_path: str | Path | None = None) -> List[str]:
    models = _load_models_yaml(config_path)
    return [m["id"] for m in models if m.get("id") and m.get("active", False)]


def load_model_sizes(config_path: str | Path | None = None) -> Dict[str, str]:
    registry = load_model_registry(config_path)
    return {mid: str(meta.get("size", "unknown")) for mid, meta in registry.items()}


def load_model_categories(config_path: str | Path | None = None) -> Dict[str, str]:
    registry = load_model_registry(config_path)
    return {mid: str(meta.get("category", "unknown")) for mid, meta in registry.items()}


# Backward-compatible module-level maps
model_sizes = load_model_sizes()
model_categories = load_model_categories()
