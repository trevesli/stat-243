import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ars')))

import pytest
import numpy as np
import matplotlib.pyplot as plt
from ars.sampler import (
    ars,
    construct_envelope,
    sample_piecewise_linear
)

#################################################
### Mock data and helper functions for testing
#################################################
@pytest.fixture
def mock_envelope_data():
    x = np.linspace(0, 10, 100)
    h = lambda x: np.exp(-x)  # Define h as a callable function
    return x, h

@pytest.fixture
def mock_piecewise_linear_data():
    x = np.array([0, 2, 4, 6, 8, 10])
    y = np.array([1, 0.8, 0.6, 0.4, 0.2, 0.1])
    return x, y

def log_concave(x):
    return np.exp(-x**2)

def gaussian(x):
    return np.exp(-0.5 * x**2)

#################################################
### Unit tests
#################################################
def test_inputs():
    with pytest.raises(TypeError, match="'domain' must be a tuple."):
        ars(log_concave, 100, domain="invalid_domain")
    with pytest.raises(ValueError, match="'domain' must contain exactly two elements."):
        ars(log_concave, 100, domain=(-5,))
    with pytest.raises(ValueError, match="'domain' must contain exactly two elements."):
        ars(log_concave, 100, domain=())
    with pytest.raises(ValueError, match="'domain' must contain exactly two elements."):
        ars(log_concave, 100, domain=(-5, 5, 10))
    with pytest.raises(TypeError, match="'domain' must be a tuple."):
        ars(log_concave, 100, domain=[-5, 5])
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        ars(gaussian, num_samples=-100, domain=(-5, 5))
    with pytest.raises(ValueError, match="Invalid domain: values must be numeric."):
        ars(log_concave, 100, domain=(-np.inf, "not_a_number"))
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        ars(log_concave, -5, domain=(-5,5))
    with pytest.raises(ValueError, match="'domain_threshold' must be a positive number."):
        ars(log_concave, 100, domain=(-5, 5), domain_threshold="invalid")
    with pytest.raises(ValueError, match="'domain_threshold' must be a positive number."):
        ars(log_concave, 100, domain=(-5, 5), domain_threshold=-1)
    with pytest.raises(ValueError, match="'domain_step' must be a positive number."):
        ars(log_concave, 100, domain=(-5, 5), domain_step="invalid")
    with pytest.raises(ValueError, match="'domain_step' must be a positive number."):
        ars(log_concave, 100, domain=(-5, 5), domain_step=-0.1)
    with pytest.raises(ValueError, match="'init_threshold' must be a positive number."):
        ars(log_concave, 100, domain=(-5, 5), init_threshold="invalid")
    with pytest.raises(ValueError, match="'init_threshold' must be a positive number."):
        ars(log_concave, 100, domain=(-5, 5), init_threshold=-0.1)
    with pytest.raises(ValueError, match="'max_step' must be a positive integer."):
        ars(log_concave, 100, domain=(-5, 5), max_step="invalid")
    with pytest.raises(ValueError, match="'max_step' must be a positive integer."):
        ars(log_concave, 100, domain=(-5, 5), max_step=-10)
    with pytest.raises(ValueError, match="Lower bound of 'domain' must be less than upper bound."):
        ars(log_concave, 100, domain=(5, -5))
    with pytest.raises(ValueError, match="Lower bound of 'domain' must be less than upper bound."):
        ars(log_concave, 100, domain=(5, 5))
    with pytest.raises(TypeError, match="Must use a callable function 'f'."):
        ars(f="not_callable", num_samples=100, domain=(-5, 5))
    with pytest.raises(ValueError, match="'burn_in' must be a non-negative integer."):
        ars(log_concave, 100, domain=(-5, 5), burn_in=-1)
    with pytest.raises(ValueError, match="'burn_in' must be a non-negative integer."):
        ars(log_concave, 100, domain=(-5, 5), burn_in=1.5)
    with pytest.raises(ValueError, match="'num_init_points' must be an integer >= 3."):
        ars(log_concave, 100, domain=(-5, 5), num_init_points=2)
    with pytest.raises(ValueError, match="'num_init_points' must be an integer >= 3."):
        ars(log_concave, 100, domain=(-5, 5), num_init_points=3.5)

def test_basic():
    num_samples = 100
    domain = (-5,5)
    try:
        samples = ars(log_concave, num_samples, domain=(-5,5))
        print(f"Generated {len(samples)} samples.")
        assert len(samples) == num_samples
        assert np.all(samples >= domain[0]) and np.all(samples <= domain[1]) # samples are in domain
    except ValueError as e:
        print(f"Error: {e}")

def test_burn_in():
    num_samples = 100
    domain = (-5, 5)
    burn_in = 20
    samples = ars(log_concave, num_samples, domain, burn_in=burn_in)
    
    assert len(samples) >= num_samples

def test_adaptive_domain_search():
    num_samples = 100
    domain = (-np.inf, np.inf)  # Use adaptive domain search
    samples = ars(log_concave, num_samples, domain)
    
    assert len(samples) == num_samples

def test_envelope_and_sampling(mock_envelope_data, mock_piecewise_linear_data):
    # Test construct_envelope
    x, h = mock_envelope_data
    domain = (min(x), max(x))  # Define the domain based on the input data
    envelope = construct_envelope(x, h, domain)

    # Validate envelope construction
    assert isinstance(envelope, tuple), "Envelope should be a tuple."
    assert len(envelope) == 2, "Envelope tuple should contain two elements (pieces, z_points)."
    pieces, z_points = envelope

    # Validate pieces
    assert isinstance(pieces, list), "Pieces should be a list."
    assert all(isinstance(item, tuple) and len(item) == 2 for item in pieces), \
        "Each piece should be a tuple of (slope, intercept)."

    # Validate z_points
    assert isinstance(z_points, list), "z_points should be a list."
    assert len(z_points) == len(pieces) + 1, \
        f"Expected {len(pieces) + 1} z_points, but got {len(z_points)}."

    # Validate sampling (mock_piecewise_linear_data test)
    x, y = mock_piecewise_linear_data
    sampled = sample_piecewise_linear(pieces, z_points)

    # Check sampled is scalar value
    assert isinstance(sampled, (float, np.float64)), "Sampled point should be a scalar value."
    assert sampled >= min(x) and sampled <= max(x), "Sampled point should be within the x range."

def test_construct_envelope_basic():
    """
    Checks that the function calculates the correct number of line segments
    (pieces) and intersection points (z_points) based on hull points.
    """
    hull_points = np.array([-3, 0, 3])
    h = lambda x: -0.5 * x**2  # Log of Gaussian
    domain = (-5, 5)

    # Construct envelope
    pieces, z_points = construct_envelope(hull_points, h, domain)

    # Check first and last points in z_points are domain boundaries
    assert np.isclose(z_points[0], domain[0]), \
        f"First z_point should be {domain[0]}, but got {z_points[0]}."
    assert np.isclose(z_points[-1], domain[1]), \
        f"Last z_point should be {domain[1]}, but got {z_points[-1]}."

    # Additional check that the z_points are in increasing order
    for i in range(1, len(z_points)):
        assert z_points[i] > z_points[i - 1], \
            f"z_points should be in increasing order, but {z_points[i-1]} > {z_points[i]}."

    # Check slopes and intercepts are of expected type and value
    for piece in pieces:
        assert isinstance(piece, tuple), "Each piece should be a tuple."
        assert len(piece) == 2, "Each piece should contain (slope, intercept)."
        assert isinstance(piece[0], float), "Slope should be a float."
        assert isinstance(piece[1], float), "Intercept should be a float."

def test_construct_envelope_invalid():
    hull_points = [1, 2]  # Less than 3 points
    h = lambda x: np.log(x)
    domain = (0, 10)
    with pytest.raises(ValueError):
        construct_envelope(hull_points, h, domain)

def test_stability():
    # Gaussian with small variance
    small_variance_gaussian = lambda x: np.exp(-0.5 * (x / 0.01)**2)
    domain = (-0.05, 0.05)  
    samples = ars(small_variance_gaussian, num_samples=500, domain=domain, num_init_points=10)
    assert np.all(samples >= domain[0]) and np.all(samples <= domain[1])

    # Gaussian with large variance
    large_variance_gaussian = lambda x: np.exp(-0.5 * (x / 10)**2)
    domain = (-50, 50)  # Larger domain
    samples = ars(large_variance_gaussian, num_samples=500, domain=domain, num_init_points=10)
    assert np.all(samples >= domain[0]) and np.all(samples <= domain[1])

def test_sampling_dist():
    # Validate that sampled distribution approximates true density.
    domain = (-5, 5) 
    samples = ars(gaussian, num_samples=10000, domain=domain, num_init_points=10)

    hist, bin_edges = np.histogram(samples, bins=50, density=True) 
    # Compare histogram to the true (Gaussian) density 
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    true_density = np.exp(-0.5 * bin_centers**2) / np.sqrt(2 * np.pi) # normalised

    assert np.allclose(hist, true_density, atol=0.05)