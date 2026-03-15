"""utils/config.py — Single config loader for the whole project."""
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def load_config(path: str | None = None) -> dict:
    """
    Load config.toml. Also loads .env from the project root if python-dotenv
    is installed (silently skipped if not).

    Parameters
    ----------
    path : str | None
        Explicit path to a .toml file. Defaults to <project_root>/config.toml.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    if path is None:
        path = str(_ROOT / "config.toml")

    with open(path, "rb") as f:
        return tomllib.load(f)
