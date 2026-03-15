import os
import tempfile
import textwrap
from pathlib import Path
from utils.config import load_config


def _write_toml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False)
    f.write(textwrap.dedent(content))
    f.flush()
    return f.name


def test_loads_toml_values():
    path = _write_toml("""
        [sailing]
        window_start = "08:00"
        window_end   = "16:00"
        wind_speed_min = 2.0
    """)
    cfg = load_config(path)
    assert cfg["sailing"]["window_start"] == "08:00"
    assert cfg["sailing"]["wind_speed_min"] == 2.0


def test_returns_dict():
    path = _write_toml("[model]\nn_estimators = 300\n")
    cfg = load_config(path)
    assert isinstance(cfg, dict)


def test_loads_project_config():
    """Smoke test: project config.toml loads without error."""
    cfg = load_config()
    assert "sailing" in cfg
    assert "prediction" in cfg
