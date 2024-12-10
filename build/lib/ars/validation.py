import numpy as np

def is_log_concave(f, x_range, eps=1e-10):
    """Checks if a function is log-concave over the given range."""
    x = np.asarray(x_range)
    
    if np.any(np.diff(x) <= 0): # Ensure no duplicates in range (for numerical differentiation)
        raise ValueError("x_range need to be increasing.")
    
    f_values = f(x)
    if np.any(f_values <= 0):
        raise ValueError("Function values must be positive.")
    
    f_values = np.maximum(f_values, eps)
    #f_prime = np.gradient(f_values, x)
    #log_deriv = f_prime / f_values
    #is_descending = np.all(np.diff(log_deriv) <= 0)
    
    log_f_values = np.log(f_values)
    
    # Compute second derivative
    log_f_second_derivative = np.gradient(np.gradient(log_f_values, x), x)
    
    # Check second derivative is non-positive (i.e. log-concave condition)
    is_log_concave = np.all(log_f_second_derivative <= 0)
    
    return is_log_concave