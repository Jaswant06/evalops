"""EvalOps: evaluation and release gating for LLM applications."""

__version__ = "0.1.0"

# Bump whenever deterministic checks, metric definitions or judge parsing change.
# It is part of every run fingerprint, so results scored by different logic are
# never mistaken for each other.
SCORING_VERSION = "3"
