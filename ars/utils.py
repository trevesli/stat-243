import numpy as np

# No longer used
def check_overflow_underflow(values, dtype=np.float64):
    """Checks if values are at risk of overflow/underflow, and clip them."""
    min_value = np.finfo(dtype).tiny  # Smallest positive number representable in the dtype
    max_value = np.finfo(dtype).max   # Largest number representable in the dtype
    
    values = np.clip(values, min_value, max_value)

    return values

def h_log(f, x):
    """Manual caching for log of the target function with underflow protection."""
    if isinstance(x, np.ndarray):
        x_key = tuple(x.tolist())  # Convert array to hashable tuple
    else:
        x_key = x

    if x_key not in h_log.cache:
        f_value = f(x)
        f_value = check_overflow_underflow(f_value)
        if isinstance(f_value, np.ndarray):
            f_value = np.maximum(f_value, np.finfo(float).eps)  # Prevent underflow for arrays
        else:
            f_value = max(f_value, np.finfo(float).eps)  # Prevent underflow for scalars
        h_log.cache[x_key] = np.log(f_value)

    return h_log.cache[x_key]

def h_log_test(f, x):
    """Manual caching for log of the target function."""
    # Checks
    if f is None:
        raise ValueError("Input must not be empty")
    if not callable(f):
        raise TypeError("Input must be callable")
    if isinstance(x, np.ndarray):
        x_key = tuple(x.tolist())  # Convert array to hashable tuple
    else:
        x_key = x
    
    # Ensure cache is initialized (if not done already)
    if not hasattr(h_log, "cache"):
        h_log.cache = {}
    
    # Check cache and compute value if not already computed
    if x_key not in h_log.cache:
        try:
            f_value = f(x)
            # Caching the result
            h_log.cache[x_key] = f_value
        except Exception as e:
            raise ValueError(f"Error in computing function value: {e}")
    
    return h_log.cache[x_key]

h_log.cache = {}

h_cache = {}

# No longer used
def h_cached(f, x):
    """Cache h values to avoid recomputing for the same x values."""
    x_key = tuple(x.tolist()) if isinstance(x, np.ndarray) else x

    if x_key not in h_cache:
        h_cache[x_key] = h_log(f, x)
    return h_cache[x_key]