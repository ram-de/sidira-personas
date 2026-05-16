"""Layer B — Group Encoder. Generic. Reduces each group to a fixed-width embedding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def encode_group(features: pd.DataFrame, dim: int) -> tuple[np.ndarray, PCA | None]:
    if features.shape[1] == 0:
        return np.zeros((len(features), dim)), None
    n_components = min(dim, features.shape[1], max(1, features.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=42)
    z = pca.fit_transform(features.values)
    if z.shape[1] < dim:
        pad = np.zeros((z.shape[0], dim - z.shape[1]))
        z = np.hstack([z, pad])
    return z, pca


def encode_all(group_features: dict[str, pd.DataFrame], dim: int) -> tuple[np.ndarray, dict]:
    embeddings, models = [], {}
    for name, feats in group_features.items():
        z, pca = encode_group(feats, dim)
        embeddings.append(z)
        models[name] = pca
    return np.hstack(embeddings), models
