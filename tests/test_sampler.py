import pytest
import numpy as np
import matplotlib.pyplot as plt
from ars.sampler import (
    ars,
    construct_envelope
)

def log_concave(x):
    return np.exp(-x**2)

def gaussian(x):
    return np.exp(-0.5 * x**2)

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
    true_density = np.exp(-0.5 * bin_centers**2)

    assert np.allclose(hist, true_density, atol=0.05)

    # Debug plot
    #plt.plot(bin_centers, true_density, label="True Density", color="red")
    #plt.hist(samples, bins=50, density=True, alpha=0.6, label="Sampled Distribution")
    #plt.legend()
    #plt.show()