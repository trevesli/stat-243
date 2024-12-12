import numpy as np

def is_log_concave(f, x_range):
    """
    Check if the function f is log-concave over the x_range.
    
    Args:
        f(function): Probability density function (maybe unnormalized)
        x_range(array_like): Range of x values to check
    
    Returns: 
        bool: Whether the function f is concave or not.
    """
    x = np.asarray(x_range)
    f_values = f(x)
    
    if np.any(f_values <= 0):
        raise ValueError("Function values must be positive.")
    
    log_f_values = np.log(f_values)
    
    # Compute second derivative
    log_f_second_derivative = np.gradient(np.gradient(log_f_values, x), x)
    
    # Check second derivative is non-positive (i.e. log-concave condition)
    is_log_concave = np.all(log_f_second_derivative <= 0)
    
    return is_log_concave



