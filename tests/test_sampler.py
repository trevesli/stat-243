import pytest
import numpy as np
from ars import ars

def test_ars():
    """
    Test ars.

    Verifies that ARS generates the correct number of samples 
    Ensures that samples fall within the specified domain.
    """
    # Parameters for ARS algorithm
    num_samples = 1000
    x_init = [-2, 0, 2]  # Initial points for algorithm
    
    # Do sampling
    samples = ars(gaussian, num_samples, x_init)
    
    # Validate outputs
    assert len(samples) == num_samples  # Check correct number of samples
    assert np.all(samples >= -10) and np.all(samples <= 10)  # Check domain bounds

def test_edge_case_distributions():
    """
    Test edge-case distributions.
    """
    # Uniform dist (log-concave)
    uniform = lambda x: 1 if -5 <= x <= 5 else 0
    samples = ars(uniform, num_samples=500, x_init=[-4, 0, 4])
    assert np.all(samples >= -5) and np.all(samples <= 5)  # All samples within domain
    
    # Exponential dist (log-concave)
    exponential = lambda x: np.exp(-x) if x >= 0 else 0
    samples = ars(exponential, num_samples=500, x_init=[0.5, 1, 2])
    assert np.all(samples >= 0)  # All samples in valid range

def test_numerical_stability():
    """
    Test numerical stability for very small or very large values.
    """
    # Gaussian with small variance
    small_variance_gaussian = lambda x: np.exp(-0.5 * (x / 0.01)**2)
    samples = ars(small_variance_gaussian, num_samples=500, x_init=[-0.02, 0, 0.02])
    assert np.all(samples >= -0.05) and np.all(samples <= 0.05)  # Domain of interest

    # Gaussian with large variance
    large_variance_gaussian = lambda x: np.exp(-0.5 * (x / 10)**2)
    samples = ars(large_variance_gaussian, num_samples=500, x_init=[-20, 0, 20])
    assert np.all(samples >= -50) and np.all(samples <= 50)  # Larger range

def test_sampling_distribution():
    """
    Validate that the sampled distribution approximates the true density.
    """
    # Gaussian distribution
    sampled = ars(gaussian, num_samples=10000, x_init=[-3, 0, 3])
    hist, bin_edges = np.histogram(sampled, bins=50, density=True)
    
    # Compare histogram to true density
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    true_density = np.exp(-0.5 * bin_centers**2)
    assert np.allclose(hist, true_density, atol=0.05)  # Allow for stochastic variability