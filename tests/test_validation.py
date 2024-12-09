import pytest
import numpy as np
from ars import is_log_concave, ars

def test_is_log_concave():
    """
    Test is_log_concave.

    Checks that the function correctly identifies log-concave functions
    Raise ValueError if not, or if functions are non-positive
    """
    # Test with Gaussian (valid)
    x_range = np.linspace(-5, 5, 100)
    assert is_log_concave(gaussian, x_range) == True

    # Test with non-valid (quadratic; non-log-concave)
    non_log_concave = lambda x: x**2
    with pytest.raises(ValueError):
        is_log_concave(non_log_concave, x_range)

def test_invalid_inputs():
    """
    Test appropriate errors are raised for invalid inputs.
    """
    # Define non-log-concave function
    non_log_concave = lambda x: x**2 + 1
    
    # Check if function raises ValueError for non-log-concave densities
    with pytest.raises(ValueError):
        ars(non_log_concave, num_samples=100, x_init=[-1, 0, 1])
    
    # Check for invalid number of samples
    with pytest.raises(ValueError):
        ars(gaussian, num_samples=-100, x_init=[-1, 0, 1])
    
    # Check for insufficient initial points
    with pytest.raises(ValueError):
        ars(gaussian, num_samples=100, x_init=[0])

def test_log_concavity_checks():
    """
    Test that is_log_concave correctly detects issues during execution.
    """
    # Non-log-concave function (exceeds bounds during runtime)
    problematic_func = lambda x: np.exp(np.sin(x))
    with pytest.raises(ValueError):
        ars(problematic_func, num_samples=500, x_init=[-2, 0, 2])