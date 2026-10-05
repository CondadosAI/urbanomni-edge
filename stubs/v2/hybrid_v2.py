"""Inference-only stand-in for upstream v2/hybrid_v2.py.

The released checkpoints pickle a ``HybridPoseModel26``. Upstream, that class only
changes the training loss; the forward pass is the stock YOLO26 pose model. This
stub lets the checkpoint unpickle without importing any upstream code.
"""
from ultralytics.nn.tasks import PoseModel


class HybridPoseModel26(PoseModel):
    pass
