import json
import csv
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_from_json(filepath: str) -> list:
    """Load a list of alpha dicts from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} alphas from {filepath}")
    return data


def load_from_csv(filepath: str) -> list:
    """
    Load alphas from a CSV file.

    Expected columns: name, expression
    Additional columns are included as-is in the dict.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    alphas = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alphas.append(dict(row))
    logger.info(f"Loaded {len(alphas)} alphas from {filepath}")
    return alphas


# Platform default settings (from Brain UI: USA/D1/TOP3000)
_NEUTRALIZATION_OPTIONS = ["SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET"]
_DECAY_OPTIONS          = [2, 4, 6, 8]     # cluster around platform default of 4
_TRUNCATION_OPTIONS     = [0.07, 0.08, 0.09]  # tight range around 0.08


def build_simulation_payload(
    expression: str,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = None,
    decay: int = None,
    truncation: float = None,
    delay: int = 1,
    alpha_type: str = "REGULAR",
    randomize_settings: bool = True,
) -> dict:
    """
    Build the Brain API simulation request payload.

    Parameters
    ----------
    expression : str
        The alpha expression string, e.g. 'rank(close)'.
    randomize_settings : bool
        If True, vary neutralization/decay/truncation slightly around platform
        defaults so submissions don't look like a bot using identical settings.
    """
    # Use platform defaults (matching Brain UI), with optional human-like variation
    if randomize_settings:
        neu   = neutralization or random.choice(_NEUTRALIZATION_OPTIONS)
        dec   = decay          if decay is not None else random.choice(_DECAY_OPTIONS)
        trunc = truncation     if truncation is not None else random.choice(_TRUNCATION_OPTIONS)
    else:
        neu   = neutralization or "SUBINDUSTRY"
        dec   = decay          if decay is not None else 4
        trunc = truncation     if truncation is not None else 0.08

    return {
        "type": alpha_type,
        "settings": {
            "instrumentType": instrument_type,
            "region": region,
            "universe": universe,
            "delay": delay,
            "decay": dec,
            "neutralization": neu,
            "truncation": trunc,
            "unitHandling": "VERIFY",
            "pasteurization": "ON",
            "language": "FASTEXPR",
            "visualization": False,
            "nanHandling": "OFF",
        },
        "regular": expression,
    }
