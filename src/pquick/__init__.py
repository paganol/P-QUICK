"""P-QUICK package."""

from .config import PipelineConfig, load_config


def run_pipeline(config: PipelineConfig):
	"""Lazy wrapper to avoid importing pipeline at package import time."""
	from .pipeline import run_pipeline as _run_pipeline

	return _run_pipeline(config)

__all__ = ["PipelineConfig", "load_config", "run_pipeline"]
