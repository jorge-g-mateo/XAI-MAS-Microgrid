"""Extra quality indicators not shipped (or not general enough) in jMetalPy.

Currently: the Spread / dispersion indicator (Delta), the pure *diversity*
quantifier for Pareto-front quality indicators
(convergence: epsilon, GD / diversity: Spread / both: IGD, HV). Lower is
better: 0 means the candidate front is perfectly evenly spaced and reaches the
extremes of the reference front.

For two objectives we use the canonical definition (Deb's Delta):

    Delta = (e1 + e2 + sum_i |d_i - d_mean|) / (e1 + e2 + (v-1) * d_mean)

where the d_i are the distances between *consecutive* points of the candidate
front sorted along the front, and e1/e2 the distances between the extreme
points of the reference front and of the candidate front.

The consecutive ordering only exists for m = 2. For m >= 3 we apply the
standard nearest-neighbour generalization (Wang et al.): d_i is the distance
from point i to its nearest neighbour in the candidate front, and one extreme
term e_k per objective (distance between the reference front's best point and
the candidate front's best point on objective k):

    Delta = (sum_k e_k + sum_i |d_i - d_mean|) / (sum_k e_k + v * d_mean)

Distances are computed on raw objective values, consistent with how the
jMetalPy GD/IGD/epsilon indicators are used elsewhere in this module.
"""

from __future__ import annotations

import numpy as np


def _pairwise_dist(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def spread(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Spread / dispersion indicator Delta of ``front`` w.r.t. ``reference_front``.

    Both arrays are (n_points, m) in *minimization* objective space (the raw
    ``solution.objectives`` of jMetal). Lower is better.
    """
    front = np.asarray(front, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    if front.ndim != 2 or len(front) < 2:
        return float("nan")
    m = front.shape[1]
    v = len(front)

    if m == 2:
        # Sort along the front (by first objective) -> consecutive distances.
        order = np.argsort(front[:, 0])
        f = front[order]
        d = np.sqrt(((f[1:] - f[:-1]) ** 2).sum(axis=1))
        # Extremes: best point per objective of reference vs candidate front.
        e = 0.0
        for k in range(2):
            ref_ext = ref[np.argmin(ref[:, k])]
            f_ext = f[np.argmin(f[:, k])]
            e += np.linalg.norm(ref_ext - f_ext)
        d_mean = d.mean()
        denom = e + (v - 1) * d_mean
        return float((e + np.abs(d - d_mean).sum()) / denom) if denom > 0 else 0.0

    # m >= 3: nearest-neighbour generalization.
    dist = _pairwise_dist(front)
    np.fill_diagonal(dist, np.inf)
    d = dist.min(axis=1)                       # distance to nearest neighbour
    e = 0.0
    for k in range(m):
        ref_ext = ref[np.argmin(ref[:, k])]
        f_ext = front[np.argmin(front[:, k])]
        e += np.linalg.norm(ref_ext - f_ext)
    d_mean = d.mean()
    denom = e + v * d_mean
    return float((e + np.abs(d - d_mean).sum()) / denom) if denom > 0 else 0.0
