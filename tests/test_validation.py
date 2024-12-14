import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ars')))

import pytest
import numpy as np
from ars.sampler import ars
from ars.validation import is_log_concave

def log_concave(x):
    return np.exp(-x**2)

def non_log_concave(x):
    return x**2

def gaussian(x):
    return np.exp(-0.5 * x**2)

def test_is_log_concave():
    x_range = np.linspace(-5, 5, 100)
    
    # Test with a valid log-concave function (Gaussian)
    assert is_log_concave(gaussian, x_range) == True

    # Test with non-valid (quadratic; non-log-concave)
    with pytest.raises(ValueError, match="The function is not log-concave!"):
        is_log_concave(non_log_concave, x_range)
    
    # TODO
    # Test with log-normal (valid)
    def log_normal(x):
        return np.where(x > 0, np.exp(-np.log(x)**2 / 2) / (x * np.sqrt(2 * np.pi)), 0)
    x_range_log_normal = np.linspace(0.1, 5, 100)
    assert is_log_concave(log_normal, x_range_log_normal) == True

    # Test with exponential (valid)
    def exponential(x):
        return np.exp(-x)
    x_range_exponential = np.linspace(0, 5, 100)
    assert is_log_concave(exponential, x_range_exponential) == True
    
    # Test with constant function (valid, log-concave, but wrong input type)
    constant_invalid = lambda x: 1
    x_range_constant = np.linspace(-5, 5, 100)
    with pytest.raises(ValueError):
        is_log_concave(constant_invalid, x_range_constant)

    # Test with constant function (valid, log-concave)
    constant = lambda x: np.ones_like(x)
    x_range_constant = np.linspace(-5, 5, 100)
    assert is_log_concave(constant, x_range_constant) == True

    # Test with negative values (non-log-concave)
    negative_function = lambda x: -np.exp(-x**2 / 2)
    with pytest.raises(ValueError):
        is_log_concave(negative_function, x_range)

    # Test with an edge case of an undefined function (non-log-concave)
    undefined_function = lambda x: 1 / (x**2 - 1)  # Has asymptotes
    with pytest.raises(ValueError):
        is_log_concave(undefined_function, np.linspace(-5, 5, 100))

    # Test with other non-log-concave function
    bad_function = lambda x: np.exp(np.sin(x))
    with pytest.raises(ValueError):
        ars(bad_function, num_samples=500, domain=(-5, 5))

    # Check error is raised if function values are non-positive
    non_positive_values = lambda x: -np.exp(-x**2 / 2)  # Negative values for Gaussian
    with pytest.raises(ValueError):
        is_log_concave(non_positive_values, x_range)

    # Test with log-concave function "like" Gaussian
    assert is_log_concave(lambda x: np.exp(-0.5 * (x-2)**2), np.linspace(-5, 5, 100)) == True