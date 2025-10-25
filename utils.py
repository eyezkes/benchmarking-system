# utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Literal
import pandas as pd

PathLike = Union[str, Path]

def load_dataset(path: PathLike) -> pd.DataFrame:
    """
    Load a dataset from CSV / Parquet / JSONL by file extension.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    ext = p.suffix.lower()
    if ext in {".parquet"}:
        return pd.read_parquet(p)
    if ext in {".csv"}:
        return pd.read_csv(p)
    if ext in {".jsonl", ".json"}:
        # Assumes JSON Lines for .jsonl (lines=True); for .json uses default
        return pd.read_json(p, lines=(ext == ".jsonl"))

    raise ValueError(f"Unsupported dataset extension: {ext}")

def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str]
) -> None:
    """
    Ensure required columns exist. Raises ValueError if any are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                         f"Available: {list(df.columns)}")

def sample_dataset(
    path: PathLike,
    sample_size: int,
    seed: Optional[int] = None,
    required_columns: Optional[Iterable[str]] = None,
    *,
    replace: bool = False
) -> pd.DataFrame:
    """
    Load a dataset and return a deterministic sample.
    - path: csv/parquet/json(l)
    - sample_size: number of rows to sample
    - seed: random_state passed to pandas for reproducibility
    - required_columns: if provided, validate these columns exist
    - replace: whether to sample with replacement
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")

    df = load_dataset(path)

    if required_columns:
        validate_required_columns(df, required_columns)

    if len(df) == 0:
        raise ValueError("Dataset is empty; cannot sample.")

    if not replace and sample_size > len(df):
        raise ValueError(
            f"sample_size ({sample_size}) > dataset size ({len(df)}). "
            "Use replace=True if you need more rows than available."
        )

    return df.sample(n=sample_size, random_state=seed, replace=replace).reset_index(drop=True)
