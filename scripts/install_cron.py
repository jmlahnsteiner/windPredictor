#!/usr/bin/env python3
"""
scripts/install_cron.py — Manage cron jobs for automated wind-predictor deployment.

Reads snapshot times from config.toml and creates one cron entry per snapshot,
so the full pipeline (download → stitch → predict → render → push) runs
automatically at each configured forecast checkpoint.

Usage:
    python scripts/install_cron.py          # install / update cron jobs
    python scripts/install_cron.py --list   # preview without changing anything
    python scripts/install_cron.py --remove # remove all wind-predictor cron jobs
"""

import argparse
import os
import subprocess
import sys
import tomllib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MARKER = "# wind-predictor-deploy"
_LOG_DIR = os.path.join(_ROOT, "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "deploy.log")


def _load_snapshots() -> list[str]:
    with open(os.path.join(_ROOT, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    return cfg["prediction"]["snapshots"]


def _python_exe() -> str:
    venv = os.path.join(_ROOT, ".venv", "bin", "python")
    return venv if os.path.exists(venv) else sys.executable


def _build_cron_lines() -> list[str]:
    python = _python_exe()
    deploy = os.path.join(_ROOT, "deploy.py")
    lines = []
    for snap in _load_snapshots():
        h, m = map(int, snap.split(":"))
        cmd = (
            f"cd {_ROOT} && {python} {deploy} "
            f">> {_LOG_FILE} 2>&1"
        )
        line = f"{m} {h} * * * {cmd} {_MARKER}"
        lines.append(line)
    return lines


def _get_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    # Exit code 1 with "no crontab" message is normal on a fresh system
    if result.returncode not in (0, 1):
        result.check_returncode()
    return result.stdout


def _set_crontab(content: str) -> None:
    subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def _strip_marker(crontab: str) -> str:
    return "\n".join(
        line for line in crontab.splitlines() if _MARKER not in line
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage wind-predictor cron jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list",   action="store_true", help="Preview cron lines, no changes")
    group.add_argument("--remove", action="store_true", help="Remove all wind-predictor cron jobs")
    args = parser.parse_args()

    new_lines = _build_cron_lines()

    if args.list:
        print("Cron lines to install:\n")
        for line in new_lines:
            print(" ", line)
        print(f"\nLog file: {_LOG_FILE}")
        print(f"Python  : {_python_exe()}")
        return

    existing = _get_crontab()
    cleaned = _strip_marker(existing)

    if args.remove:
        _set_crontab(cleaned + "\n" if cleaned else "")
        print(f"Removed all wind-predictor cron jobs ({_MARKER}).")
        return

    # Install ─────────────────────────────────────────────────────────────────
    os.makedirs(_LOG_DIR, exist_ok=True)

    updated = (cleaned + "\n" if cleaned else "") + "\n".join(new_lines) + "\n"
    _set_crontab(updated)

    print(f"Installed {len(new_lines)} cron job(s):\n")
    for line in new_lines:
        print(" ", line)
    print(f"\nLog file : {_LOG_FILE}")
    print(f"Python   : {_python_exe()}")
    print("\nTo remove: python scripts/install_cron.py --remove")


if __name__ == "__main__":
    main()
