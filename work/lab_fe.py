"""Adapter so the holdout harness exercises the production feature code itself."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fe


def ctx_fn(tr):
    return fe.fit_context(tr)


def build(d, ctx):
    return fe.build_features(d, ctx)
