import numpy as np
import pytest
from ars import is_log_concave, construct_envelope, calculate_envelope, ars


###############################
# "BASIC" TESTS
###############################

# Define a simple log-concave function for testing
def gaussian(x):
    """Test Gaussian function."""
    return np.exp(-0.5 * x**2)

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


###############################
# ADDITIONAL TESTS
###############################

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

def test_vectorization():
    """
    Test for support of vectorised density functions.
    """
    vectorized_gaussian = lambda x: np.exp(-0.5 * x**2)
    samples = ars(vectorized_gaussian, num_samples=1000, x_init=[-1, 0, 1])
    assert len(samples) == 1000  # Correct number of samples
    assert np.all(np.isfinite(samples))  # No invalid (NaN or Inf) samples

def test_log_concavity_checks():
    """
    Test that is_log_concave correctly detects issues during execution.
    """
    # Non-log-concave function (exceeds bounds during runtime)
    problematic_func = lambda x: np.exp(np.sin(x))
    with pytest.raises(ValueError):
        ars(problematic_func, num_samples=500, x_init=[-2, 0, 2])

def test_autodiff_support():
    """
    Test compatibility with JAX for automatic differentiation.
    """
    import jax.numpy as jnp
    from jax import grad

    # JAX Gaussian function with gradient
    jax_gaussian = lambda x: jnp.exp(-0.5 * x**2)
    grad_gaussian = grad(jax_gaussian)
    
    # Verify ARS compatibility with JAX
    samples = ars(jax_gaussian, num_samples=500, x_init=[-2, 0, 2], gradient_fn=grad_gaussian)
    assert len(samples) == 500
    assert np.all(np.isfinite(samples))

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