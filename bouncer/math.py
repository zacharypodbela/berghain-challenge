import random
import numpy as np


def generate_correlated_attributes(attribute_statistics, num_people=20000, seed=None):
    """
    Generate people with attributes following the correct marginal probabilities
    and correlations for a given scenario.

    This uses a multivariate normal copula approach:
    1. Generate correlated normal variables using the correlation matrix
    2. Transform to uniform [0,1] using normal CDF
    3. Apply marginal probabilities as thresholds
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Extract attribute names in consistent order
    attributes = sorted(attribute_statistics["relativeFrequencies"].keys())
    n_attrs = len(attributes)

    # Build correlation matrix
    corr_matrix = np.zeros((n_attrs, n_attrs))
    for i, attr_i in enumerate(attributes):
        for j, attr_j in enumerate(attributes):
            corr_matrix[i, j] = attribute_statistics["correlations"][attr_i][attr_j]

    # Generate correlated normal variables
    mean = np.zeros(n_attrs)
    z = np.random.multivariate_normal(mean, corr_matrix, size=num_people)

    # Transform to uniform via normal CDF
    from scipy.stats import norm

    u = norm.cdf(z)

    # Apply thresholds based on marginal probabilities
    people_attributes = []
    for person_idx in range(num_people):
        person_attrs = {}
        for attr_idx, attr_name in enumerate(attributes):
            threshold = attribute_statistics["relativeFrequencies"][attr_name]
            person_attrs[attr_name] = bool(u[person_idx, attr_idx] < threshold)
        people_attributes.append(person_attrs)

    return people_attributes
