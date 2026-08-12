"""
Tiny config loader shared by src/ and eval/.

Everything downstream (retrieval, attribution, grounding, generation, faithfulness) reads
model names, paths and top_k from configs/config.yaml through here rather than hardcoding
them -- that's the point of the config file, and it's what makes "switch the embedding
model and re-run" a one-line change in Month 7 instead of a grep-and-pray exercise.

Paths in the YAML are relative to the repo root, so get() resolves them against ROOT.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def get(cfg: dict, dotted_key: str, default=None):
    """cfg_get(cfg, 'retrieval.top_k') -> 5. Returns `default` for a missing key rather than
    raising, so a config written before a feature existed still loads."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(rel_path: str) -> Path:
    """Config paths are repo-root-relative; scripts get run from all over the place
    (src/ingestion/, eval/baselines/, notebooks/). Always resolve against ROOT."""
    p = Path(rel_path)
    return p if p.is_absolute() else ROOT / p
