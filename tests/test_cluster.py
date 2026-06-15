"""Guard the assign() signature change: it now also returns the full soft posterior matrix."""
from __future__ import annotations

import numpy as np

from sidira_personas.cluster import assign, fit_gmm_bic


def test_assign_returns_normalized_posteriors():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0.0, 1.0, size=(100, 4)),
        rng.normal(6.0, 1.0, size=(100, 4)),
    ])
    gmm, k = fit_gmm_bic(X, (2, 4), random_state=42)

    labels, confidence, probs = assign(gmm, X)

    assert probs.shape == (X.shape[0], k)
    # every posterior row sums to 1.0
    assert np.allclose(probs.sum(axis=1), 1.0)
    # labels are the argmax of the posterior and in range
    assert labels.min() >= 0 and labels.max() < k
    assert np.array_equal(labels, probs.argmax(axis=1))
    # confidence is the max posterior per row
    assert np.allclose(confidence, probs.max(axis=1))
