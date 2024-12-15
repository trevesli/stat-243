import pytest
import numpy as np
from scipy import stats
from unittest.mock import patch, ANY
from ars.sampler import ars
from ars.validation import is_log_concave, compare_samples_to_distribution, integrate_mean


#################################################
### Mock functions for testing
#################################################
def log_concave(x):
    return np.exp(-x**2)

def non_log_concave(x):
    return x**2

def gaussian(x):
    return np.exp(-0.5 * x**2)

def mock_target_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # Normal distribution PDF


#################################################
### Unit tests
#################################################
def test_is_log_concave():
    x_range = np.linspace(-5, 5, 100)
    
    # Test with a valid log-concave function (Gaussian)
    assert is_log_concave(gaussian, x_range) == True

    # Test with non-valid (quadratic; non-log-concave)
    with pytest.raises(ValueError, match="The function is not log-concave!"):
        is_log_concave(non_log_concave, x_range)
    
    """
    # Test with log-normal (valid)
    def log_normal(x):
        return np.where(x > 0, np.exp(-np.log(x)**2 / 2) / (x * np.sqrt(2 * np.pi)), 0)
    x_range_log_normal = np.linspace(0.1, 5, 100)
    assert is_log_concave(log_normal, x_range_log_normal) == True
    """

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

# Suppress irrelevant errors
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning"
)
# Test normal distribution samples
def test_ks_test_normal_dist():
    samples = np.random.normal(loc=0, scale=1, size=1000)
    domain = (-5, 5)

    with patch("builtins.print") as mock_print:
        with patch("scipy.stats.kstest") as mock_kstest:
            mock_kstest.return_value = (0.05, 0.9)  # KS stat and p-value (no significant diff)
            compare_samples_to_distribution(samples, lambda x: stats.norm.pdf(x), domain)
            
            mock_print.assert_any_call(ANY)  # Match any print call
            assert any("KS Statistic" in call[0][0] for call in mock_print.call_args_list)
            assert any("p-value" in call[0][0] for call in mock_print.call_args_list)
            assert any("KS Test suggests samples align well with target distribution." in call[0][0] for call in mock_print.call_args_list)

# Suppress irrelevant errors
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning"
)
# Test uniform distribution samples
def test_ks_test_uniform_dist():
    samples = np.random.uniform(low=-5, high=5, size=1000)
    domain = (-5, 5)

    with patch("builtins.print") as mock_print:
        with patch("scipy.stats.kstest") as mock_kstest:
            mock_kstest.return_value = (0.1, 0.01)  # KS stat and p-value (yes significant diff)
            compare_samples_to_distribution(samples, lambda x: stats.norm.pdf(x), domain)
            
            mock_print.assert_any_call(ANY) 
            assert any("KS Statistic" in call[0][0] for call in mock_print.call_args_list)
            assert any("p-value" in call[0][0] for call in mock_print.call_args_list)
            assert any("Warning: KS Test suggests samples differ significantly from target distribution." in call[0][0] for call in mock_print.call_args_list)

# Suppress irrelevant errors
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning"
)
# 3. Test empty sample array
def test_ks_test_empty_samples():
    samples = np.array([])
    domain = (-5, 5)

    with patch("builtins.print") as mock_print:
        compare_samples_to_distribution(samples, lambda x: stats.norm.pdf(x), domain)
        
        mock_print.assert_any_call(ANY) 
        assert any("KS Test skipped" in call[0][0] for call in mock_print.call_args_list)

# Suppress irrelevant errors
@pytest.mark.filterwarnings(
    "ignore:FigureCanvasAgg is non-interactive, and thus cannot be shown:UserWarning"
)
# Test invalid target PDF (e.g., returning NaN)
def test_ks_test_invalid_target_pdf():
    samples = np.random.normal(loc=0, scale=1, size=1000)
    domain = (-5, 5)

    # invalid target PDF returns NaN
    def invalid_target_pdf(x):
        return np.nan

    with patch("builtins.print") as mock_print:
        compare_samples_to_distribution(samples, invalid_target_pdf, domain)
        
        mock_print.assert_any_call(ANY)
        assert any("KS Test skipped" in call[0][0] for call in mock_print.call_args_list)