"""Shared paths and configuration loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ANCHOR_DATA = DATA / "anchors"
GOLDEN = DATA / "golden"
ARTIFACTS = ROOT / "artifacts" / "indexes"
RESULTS = ROOT / "results"

load_dotenv(ROOT / ".env")


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_anchors() -> dict[str, dict[str, Any]]:
    return load_yaml(CONFIGS / "anchors.yaml")["anchors"]


def load_config(name: str = "baseline") -> dict[str, Any]:
    return load_yaml(CONFIGS / f"{name}.yaml")


def apply_overrides(config: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of config with values replaced, such as {"generation": {"prompt": "p1"}}."""
    merged = copy.deepcopy(config)
    for section, values in overrides.items():
        merged[section].update(values)
    return merged
