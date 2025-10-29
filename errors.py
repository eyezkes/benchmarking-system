# errors.py
"""
Custom exception classes for the benchmarking system.

Each module (runner, model, judge, evaluator, etc.) should raise
these errors instead of generic Python exceptions. This helps
keep all error handling consistent and easy to catch.
"""

class BenchmarkError(Exception):
    """Base exception for all benchmark-related errors."""
    pass


class DatasetLoadError(BenchmarkError):
    """Raised when a dataset cannot be loaded or validated."""
    pass


class ModelError(BenchmarkError):
    """Raised when the model fails to generate a response or encounters an API issue."""
    pass


class EvaluationError(BenchmarkError):
    """Raised when a judging or evaluation step fails."""
    pass


class ConfigurationError(BenchmarkError):
    """Raised for missing or invalid configuration values."""
    pass
