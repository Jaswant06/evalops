"""DocQA: answers questions about a policy document, with section citations.

DocQA is the first application evaluated by EvalOps. It is deliberately a
separate package: it knows nothing about EvalOps, and EvalOps reaches it only
through `docqa.adapter.DocQAAdapter`.
"""

# Bump when retrieval, sectioning or prompt assembly changes; it is part of the
# version fingerprint, so cached responses from older code are not reused.
__version__ = "0.1.0"
