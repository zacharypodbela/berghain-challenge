import random
from typing import Any

import numpy as np
from scipy import linalg
from scipy.stats import norm


def binary_to_tetrachoric(binary_corr: float, p1: float, p2: float) -> float:
    """
    Convert observed binary correlation to tetrachoric (latent normal) correlation.

    Uses numerical optimization to find the tetrachoric correlation that would
    produce the observed binary correlation after thresholding.

    Args:
        binary_corr: Observed correlation between two binary variables
        p1: Marginal probability of first variable
        p2: Marginal probability of second variable

    Returns:
        Tetrachoric correlation for the latent normal variables
    """
    from scipy.optimize import brentq
    from scipy.stats import multivariate_normal

    # Handle edge cases
    if abs(binary_corr) < 1e-10:
        return 0.0
    if p1 <= 0.01 or p1 >= 0.99 or p2 <= 0.01 or p2 >= 0.99:
        # For extreme marginals, use a simple scaling approximation
        # since numerical methods become unstable
        h = 2.5  # Empirical scaling factor
        return float(np.clip(binary_corr * h, -0.99, 0.99))

    # Get normal quantiles for the marginal probabilities
    z1 = norm.ppf(p1)
    z2 = norm.ppf(p2)

    # Define function to find root: predicted_corr(rho) - observed_corr = 0
    def objective(rho: float) -> float:
        """Compute predicted binary correlation for a given tetrachoric rho."""
        if abs(rho) >= 1:
            return np.inf

        # Create bivariate normal with correlation rho
        cov = [[1, rho], [rho, 1]]

        try:
            # Compute P(X1 < z1, X2 < z2) using bivariate normal CDF
            # This is the probability that both binary variables are 1
            p11 = multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=cov)

            # Compute predicted binary correlation
            # Corr = (p11 - p1*p2) / sqrt(p1*(1-p1)*p2*(1-p2))
            numerator = p11 - p1 * p2
            denominator = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))

            if denominator > 0:
                predicted_corr = numerator / denominator
            else:
                predicted_corr = 0

            return float(predicted_corr - binary_corr)
        except Exception:
            return np.inf

    # Use root finding to solve for tetrachoric correlation
    try:
        # Search in a reasonable range
        max_search = 0.99
        min_search = -0.99

        # Check if solution exists in the range
        if objective(min_search) * objective(max_search) > 0:
            # No sign change, use approximation
            # Rough approximation: tetrachoric ≈ binary_corr * scaling_factor
            h1 = norm.pdf(z1)
            h2 = norm.pdf(z2)
            if h1 * h2 > 0:
                scaling = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)) / (h1 * h2)
                return float(np.clip(binary_corr / scaling, -0.99, 0.99))
            else:
                return float(np.clip(binary_corr * 2, -0.99, 0.99))

        # Find the root
        tetrachoric = brentq(objective, min_search, max_search, xtol=1e-6)
        return float(tetrachoric)

    except Exception:
        # If numerical method fails, fall back to approximation
        h1 = norm.pdf(z1)
        h2 = norm.pdf(z2)
        if h1 * h2 > 0:
            scaling = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)) / (h1 * h2)
            return float(np.clip(binary_corr / scaling, -0.99, 0.99))
        else:
            return float(np.clip(binary_corr * 2, -0.99, 0.99))


def make_correlation_matrix_valid(corr_matrix: np.ndarray) -> np.ndarray:
    """
    Ensure a correlation matrix is valid (symmetric, diagonal=1, positive semi-definite).

    Args:
        corr_matrix: Input correlation matrix

    Returns:
        Valid correlation matrix
    """
    # Make symmetric
    corr_matrix = (corr_matrix + corr_matrix.T) / 2

    # Set diagonal to 1
    np.fill_diagonal(corr_matrix, 1.0)

    # Make positive semi-definite using eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(corr_matrix)

    # If all eigenvalues are positive, matrix is already valid
    if np.all(eigenvalues > -1e-10):
        return corr_matrix

    # Clip negative eigenvalues to small positive value
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # Reconstruct the matrix
    corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Renormalize to ensure diagonal = 1
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d, d)
    np.fill_diagonal(corr_matrix, 1.0)

    return corr_matrix


def generate_correlated_attributes(
    attribute_statistics: dict[str, Any],
    num_people: int = 20000,
    seed: int | None = None,
) -> list[dict[str, bool]]:
    """
    Generate people with attributes following the correct marginal probabilities
    and correlations for a given scenario.

    This uses a multivariate normal copula approach with proper tetrachoric correlations:
    1. Convert observed binary correlations to tetrachoric (latent) correlations
    2. Validate and repair the correlation matrix if needed
    3. Generate correlated normal variables using the tetrachoric correlation matrix
    4. Transform to uniform [0,1] using normal CDF
    5. Apply marginal probabilities as thresholds
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Extract attribute names in consistent order
    attributes = sorted(attribute_statistics["relativeFrequencies"].keys())
    n_attrs = len(attributes)

    # Get marginal probabilities
    marginal_probs = [
        attribute_statistics["relativeFrequencies"][attr] for attr in attributes
    ]

    # Build TETRACHORIC correlation matrix (not the observed binary correlations)
    tetrachoric_matrix = np.zeros((n_attrs, n_attrs))
    for i, attr_i in enumerate(attributes):
        for j, attr_j in enumerate(attributes):
            if i == j:
                tetrachoric_matrix[i, j] = 1.0
            else:
                # Convert observed binary correlation to tetrachoric
                binary_corr = attribute_statistics["correlations"][attr_i][attr_j]
                tetrachoric_corr = binary_to_tetrachoric(
                    binary_corr, marginal_probs[i], marginal_probs[j]
                )
                tetrachoric_matrix[i, j] = tetrachoric_corr

    # Ensure the tetrachoric correlation matrix is valid
    tetrachoric_matrix = make_correlation_matrix_valid(tetrachoric_matrix)

    # Generate correlated normal variables using the TETRACHORIC correlations
    mean = np.zeros(n_attrs)
    z = np.random.multivariate_normal(mean, tetrachoric_matrix, size=num_people)

    # Transform to uniform via normal CDF
    u = norm.cdf(z)

    # Apply thresholds based on marginal probabilities
    people_attributes = []
    for person_idx in range(num_people):
        person_attrs = {}
        for attr_idx, attr_name in enumerate(attributes):
            threshold = marginal_probs[attr_idx]
            person_attrs[attr_name] = bool(u[person_idx, attr_idx] < threshold)
        people_attributes.append(person_attrs)

    return people_attributes
