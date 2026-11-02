"""Pipelines of blocks: the Workbench's engine. See graph.py."""
from .graph import Graph, PipelineError, REGISTRY, Runner, block  # noqa: F401
from . import blocks  # noqa: F401  (registers every block)
