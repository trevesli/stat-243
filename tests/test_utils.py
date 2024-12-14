import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ars')))

import pytest
import numpy as np
from ars.utils import check_overflow_underflow, h_log, h_cached

"""
def test_check_overflow_underflow():
    large_number = 1e308
    small_number = 1e-308
    normal_number = 1.0
    
    assert check_overflow_underflow(large_number) == "overflow", \
        "Large number should trigger 'overflow'."
    assert check_overflow_underflow(small_number) == "underflow", \
        "Small number should trigger 'underflow'."
    assert check_overflow_underflow(normal_number) == "normal", \
        "Normal number should not trigger overflow or underflow."
"""

"""
def test_h_cached():
    def f(x):
        return x * 2 

    x = np.array([1.0, 2.0])

    cached_results = h_cached(f, x)  # Pass the callable function `f`

    assert isinstance(cached_results, np.ndarray), "Cached result should be a numpy array."
    assert pytest.approx(cached_results, rel=1e-6) == h_log(f, x), \
        "Cached result should match the h_log output."
    
    cached_results_again = h_cached(f, x)
    assert np.array_equal(cached_results, cached_results_again), \
        "Cached result should remain consistent across calls."
"""

def test_h_log_bad_inputs():
    # Test with an empty list (if applicable for the function)
    with pytest.raises(TypeError):
        h_log(None, 0)  

    # Test with zero (if input is a logarithmic function or similar)
    def log_function(x):
        if x <= 0:
            raise ValueError("Log input must be greater than 0")
        return np.log(x)
    with pytest.raises(ValueError, match="Log input must be greater than 0"):
        h_log(log_function, 0)

    # Pass an invalid function (non-callable object)
    with pytest.raises(TypeError, match="is not callable"):
        h_log(123, 10)

# Performance testing with large input
def test_h_log_large_input():
    # Test with a very large input
    result = h_log(np.log, 1e6)
    assert result > 0, f"Expected positive result, got {result}"