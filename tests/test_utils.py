import pytest
import numpy as np
from ars import construct_envelope, calculate_envelope, sample_piecewise_linear

def test_construct_envelope():
    """
    Test construct_envelope.

    Checks that the function calculates the correct number of line segments
    (pieces) and intersection points (z_points) (based on hull points).
    """
    # Define input parameters for envelope construction
    hull_points = np.array([-3, 0, 3])
    h = lambda x: -0.5 * x**2  # Log of Gaussian
    domain = (-5, 5)
    
    # Construct envelope
    pieces, z_points = construct_envelope(hull_points, h, domain)
    
    # Validate outputs
    assert len(pieces) == len(hull_points) - 1  # One fewer segment than hull points
    assert len(z_points) == len(hull_points) + 1  # Two more z-points than hull points

def test_calculate_envelope():
    """
    Test calculate_envelope.

    Verifies that the envelope value at a given point matches the expected
    value derived from the input line segments and their parameters.
    """
    # Define input parameters for envelope construction
    hull_points = np.array([-3, 0, 3])
    h = lambda x: -0.5 * x**2  # Log of Gaussian
    domain = (-5, 5)
    pieces, z_points = construct_envelope(hull_points, h, domain)
    
    # Test envelope value at x=0
    x = 0
    y = calculate_envelope(x, pieces, z_points)
    assert np.isclose(y, h(x), atol=1e-2)  # Ensure calculated value matches expected

def test_sample_piecewise_linear():
    # Simple linear case
    pieces = [(1, 0), (0, 1)]  # Linear and constant
    z_points = [0, 1, 2]
    samples = [sample_piecewise_linear(pieces, z_points) for _ in range(1000)]
    
    # Test range of outputs
    assert all(0 <= s <= 2 for s in samples)
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        sample_piecewise_linear([(1, 0)], [0, 1, 2])  # Mismatch
    with pytest.raises(ValueError):
        sample_piecewise_linear([(1, 0)], [0, 1, 1])  # Non-increasing