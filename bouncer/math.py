from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# ---------- Public container for runtime sampling ----------


@dataclass
class SamplerParams:
    # Fixed layout
    attributes: list[
        str
    ]  # e.g., ["berlin_local","creative","techno_lover","well_connected"]
    bit_index: dict[str, int]  # {"berlin_local":0, "creative":1, ...}

    # Exact joint over 2^K states in this bit order
    probs: np.ndarray  # shape (2**K,), sum == 1

    # Vose alias sampler tables for O(1) sampling
    alias_prob: np.ndarray  # shape (2**K,)
    alias_alias: np.ndarray  # shape (2**K,), dtype=int

    # Optional diagnostics (helpful in tests, ignored at runtime)
    achieved_marginals: dict[str, float]
    achieved_correlations: dict[str, dict[str, float]]


# ---------- Helpers: states, features, alias sampler ----------


def _enumerate_states(K: int) -> np.ndarray:
    """
    Return all 2^K binary states as an array of shape (2^K, K), dtype=uint8.
    Row s is the bit-decomposition of s with LSB = attribute 0.
    """
    S = 1 << K
    states = np.zeros((S, K), dtype=np.uint8)
    for s in range(S):
        for k in range(K):
            states[s, k] = (s >> k) & 1
    return states


def _pair_index_list(K: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(K) for j in range(i + 1, K)]


def _build_feature_matrix(
    states: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Feature vector T(x) = [x_0,...,x_{K-1}, x_0x_1, x_0x_2, ..., x_{K-2}x_{K-1}]
    Returns (F, pairs) where F shape is (2^K, K + K*(K-1)//2).
    """
    S, K = states.shape
    pairs = _pair_index_list(K)
    F = np.empty((S, K + len(pairs)), dtype=np.float64)
    # singleton features
    F[:, :K] = states
    # pair features
    for c, (i, j) in enumerate(pairs, start=K):
        F[:, c] = states[:, i] * states[:, j]
    return F, pairs


def _build_alias(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vose alias method: returns (alias_prob, alias_alias).
    Each sample: k = randint(n); return k if rand() < alias_prob[k] else alias_alias[k].
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    n = p.size
    scaled = p * n

    prob = np.zeros(n, dtype=np.float64)
    alias = np.zeros(n, dtype=np.int32)

    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]

    while small and large:
        s = small.pop()
        L = large.pop()
        prob[s] = scaled[s]
        alias[s] = L
        scaled[L] = (scaled[L] + scaled[s]) - 1.0
        if scaled[L] < 1.0:
            small.append(L)
        else:
            large.append(L)

    # Whatever remains has prob 1
    for i in small + large:
        prob[i] = 1.0
        alias[i] = i

    return prob, alias


def _sample_alias(
    alias_prob: np.ndarray, alias_alias: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    k = alias_prob.size
    idx = rng.integers(0, k, size=n, endpoint=False)
    u = rng.random(n)
    take_alias = u >= alias_prob[idx]
    idx[take_alias] = alias_alias[idx[take_alias]]
    return idx


# ---------- Helpers: moments / correlations on 0/1 scale ----------


def _phi_from_moments(p_i: float, p_j: float, p_ij: float) -> float:
    var_i = p_i * (1.0 - p_i)
    var_j = p_j * (1.0 - p_j)
    denom = np.sqrt(max(var_i, 0.0) * max(var_j, 0.0))
    if denom == 0.0:
        return 0.0
    return float((p_ij - p_i * p_j) / denom)


def _frechet_bounds(p_i: float, p_j: float) -> tuple[float, float]:
    # Bounds on P(Xi=1, Xj=1)
    p11_min = max(0.0, p_i + p_j - 1.0)
    p11_max = min(p_i, p_j)
    return p11_min, p11_max


def _target_moment_vector(
    p: np.ndarray,
    phi: np.ndarray,  # φ_ij for i!=j, diagonal ignored/zero
) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """
    From marginals p[i] and binary correlations φ_ij, build the target μ vector:
       μ = [ E[x_i], E[x_i x_j] ] in the same order as _build_feature_matrix().
    Also return the p_ij table actually used (after Frechet clipping), shape (K,K).
    """
    K = p.size
    pairs = _pair_index_list(K)
    mu = np.empty(K + len(pairs), dtype=np.float64)

    # Singletons
    mu[:K] = p

    # Pairwise E[x_i x_j]
    p_ij = np.zeros((K, K), dtype=np.float64)
    for i, j in pairs:
        # Convert φ to P11
        var_i = p[i] * (1 - p[i])
        var_j = p[j] * (1 - p[j])
        denom = np.sqrt(max(var_i, 0.0) * max(var_j, 0.0))
        p11 = p[i] * p[j] + (phi[i, j] * denom if denom > 0 else 0.0)
        # Clip to Frechet bounds to avoid tiny infeasibilities
        lo, hi = _frechet_bounds(p[i], p[j])
        p11 = float(np.clip(p11, lo, hi))
        p_ij[i, j] = p11
        p_ij[j, i] = p11

    # Fill μ pair block in the same pair order
    for c, (i, j) in enumerate(pairs, start=K):
        mu[c] = p_ij[i, j]

    return mu, pairs, p_ij


def _moments_from_probs(
    probs: np.ndarray, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute E[x_i] and E[x_i x_j] from the full joint.
    Returns (p, p_ij) with shapes (K,) and (K,K).
    """
    probs = probs / probs.sum()
    S, K = states.shape
    # E[x_i]
    p = probs @ states  # (K,)
    # E[x_i x_j]
    p_ij = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            val = np.dot(probs, states[:, i] * states[:, j])
            p_ij[i, j] = val
            p_ij[j, i] = val
    return p, p_ij


def _phi_matrix_from_p_and_pij(
    p: np.ndarray, p_ij: np.ndarray, attrs: list[str]
) -> dict[str, dict[str, float]]:
    phi = {}
    for i, ai in enumerate(attrs):
        row = {}
        for j, aj in enumerate(attrs):
            row[aj] = float(
                _phi_from_moments(p[i], p[j], p_ij[i, j]) if i != j else 1.0
            )
        phi[ai] = row
    return phi


# ---------- Core fitter (pairwise max-entropy over 0/1) ----------


def generate_probability_vectors(attribute_statistics: dict[str, Any]) -> SamplerParams:
    """
    Fit a pairwise log-linear model P(x) ∝ exp(b·x + x^T W x / 2) on x ∈ {0,1}^K
    that matches the provided marginals and pairwise binary correlations φ (as closely as feasible).
    Returns joint probs over all 2^K states and an alias table for O(1) sampling.
    """
    # 1) Stable attribute order
    attributes = sorted(attribute_statistics["relativeFrequencies"].keys())
    bit_index = {a: i for i, a in enumerate(attributes)}
    K = len(attributes)

    # 2) Targets: p (marginals) and φ (binary correlations)
    p = np.array(
        [attribute_statistics["relativeFrequencies"][a] for a in attributes],
        dtype=np.float64,
    )
    phi = np.zeros((K, K), dtype=np.float64)
    for i, ai in enumerate(attributes):
        row = attribute_statistics["correlations"][ai]
        for j, aj in enumerate(attributes):
            if i == j:
                continue
            phi[i, j] = float(row[aj])

    # 3) Build target sufficient statistics μ = [E[x_i], E[x_i x_j]]
    mu, pairs, p_ij_target = _target_moment_vector(p, phi)

    # 4) Precompute feature matrix over all 2^K states
    states = _enumerate_states(K)  # (S,K)
    F, _ = _build_feature_matrix(states)  # (S,M)
    S, M = F.shape

    # 5) Convex dual objective: f(θ) = log Z(θ) - μ·θ, ∇ = Eθ[T] - μ
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        Fx = F @ theta  # (S,)
        logZ = logsumexp(Fx)
        probs = np.exp(Fx - logZ)  # softmax
        Et = probs @ F  # (M,)
        f = logZ - float(mu @ theta)
        g = Et - mu
        return f, g

    # 6) Initialize parameters: b_i ≈ logit(p_i), W_ij = 0
    eps = 1e-9
    logits = np.log(np.clip(p, eps, 1 - eps)) - np.log(np.clip(1 - p, eps, 1 - eps))
    theta0 = np.zeros(M, dtype=np.float64)
    theta0[:K] = logits  # good warm start

    # 7) Optimize
    res = minimize(
        fun=lambda th: objective(th)[0],
        x0=theta0,
        jac=lambda th: objective(th)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )

    theta = res.x

    # 8) Final joint and diagnostics
    Fx = F @ theta
    logZ = logsumexp(Fx)
    probs = np.exp(Fx - logZ)  # (S,)

    # diagnostics (exact from probs; no sampling needed)
    p_hat, p_ij_hat = _moments_from_probs(probs, states)

    achieved_marginals = {a: float(p_hat[i]) for i, a in enumerate(attributes)}
    achieved_correlations = _phi_matrix_from_p_and_pij(p_hat, p_ij_hat, attributes)

    # 9) Alias table for O(1) sampling
    alias_prob, alias_alias = _build_alias(probs)

    return SamplerParams(
        attributes=attributes,
        bit_index=bit_index,
        probs=probs,
        alias_prob=alias_prob,
        alias_alias=alias_alias,
        achieved_marginals=achieved_marginals,
        achieved_correlations=achieved_correlations,
    )


# ---------- O(1) runtime sampler ----------


def generate_correlated_attributes(
    params: SamplerParams, num_people: int
) -> list[dict[str, bool]]:
    """
    Sample `num_people` people from the fitted joint in O(1) each using the alias table.
    Returns a list of {attr: bool} dicts consistent with `params.attributes` bit order.
    """
    K = len(params.attributes)
    rng = np.random.default_rng()
    idx = _sample_alias(params.alias_prob, params.alias_alias, num_people, rng)

    # Map state index -> bit vector -> dict
    # Precompute all states once (tiny: at most 64x6)
    states = _enumerate_states(K)  # (2^K, K)
    bits = states[idx]  # (N, K), uint8

    people = []
    attrs = params.attributes
    for row in bits:
        d = {attrs[k]: bool(row[k]) for k in range(K)}
        people.append(d)
    return people
