# utils.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_dataset(path: PathLike) -> pd.DataFrame:
    """Load a dataset by file extension (CSV / Parquet / JSON / JSONL).

    Args:
        path: File path to load.

    Returns:
        A pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is unsupported.
        Exception: Any read error from pandas is propagated.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            df = pd.read_parquet(p)
        elif ext == ".csv":
            df = pd.read_csv(p)
        elif ext in {".jsonl", ".json"}:
            # .jsonl => lines=True; .json => default behavior
            df = pd.read_json(p, lines=(ext == ".jsonl"))
        else:
            raise ValueError(f"Unsupported dataset extension: {ext}")
    except Exception as exc:
        logger.error("Failed to read dataset %s: %s", p, exc)
        raise

    logger.info("Loaded dataset: %s (rows=%d, cols=%d)", p.name, len(df), len(df.columns))
    return df


def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Ensure all `required` columns exist in `df`.

    Args:
        df: DataFrame to validate.
        required: Column names that must be present.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available: {list(df.columns)}"
        )
    logger.debug("All required columns present: %s", list(required))


def sample_dataset(
    path: PathLike,
    sample_size: int,
    seed: Optional[int] = None,
    required_columns: Optional[Iterable[str]] = None,
    *,
    replace: bool = False,
) -> pd.DataFrame:
    """Load a dataset from disk and return a deterministic sample.

    Args:
        path: CSV/Parquet/JSON/JSONL file path.
        sample_size: Number of rows to sample (must be > 0).
        seed: Random seed passed to pandas for reproducibility.
        required_columns: Columns to validate before sampling.
        replace: Whether to sample with replacement.

    Returns:
        Sampled DataFrame (index reset).

    Raises:
        ValueError: If sample_size <= 0, dataset empty, or sample_size too large without replacement.
        FileNotFoundError / ValueError / Exception: Propagated from `load_dataset`.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    df = load_dataset(path)

    if required_columns:
        validate_required_columns(df, list(required_columns))

    if len(df) == 0:
        raise ValueError("Dataset is empty; cannot sample.")

    if not replace and sample_size > len(df):
        raise ValueError(
            f"sample_size ({sample_size}) > dataset size ({len(df)}). "
            "Use replace=True if you need more rows than available."
        )

    sampled = df.sample(n=sample_size, random_state=seed, replace=replace).reset_index(drop=True)
    logger.info(
        "Sampled %d/%d rows from %s (replace=%s, seed=%s)",
        len(sampled), len(df), Path(path).name, replace, seed
    )
    return sampled
