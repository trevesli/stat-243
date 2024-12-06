### V1 Below ###

# import numpy as np
# import scipy.stats as stats

# # Manual caching for h_log
# def h_log(f, x):
#     """Manual caching for log of the target function."""
#     if isinstance(x, np.ndarray):
#         x_key = tuple(x.tolist())  # Convert array to hashable tuple
#     else:
#         x_key = x
#     if x_key not in h_log.cache:
#         h_log.cache[x_key] = np.log(f(x))
#     return h_log.cache[x_key]

# h_log.cache = {}

# def is_log_concave(f, x_range):
#     """Checks if a function is log-concave over the given range."""
#     x = np.asarray(x_range)
#     f_values = f(x)
#     if np.any(f_values <= 0):
#         raise ValueError("Function values must be positive.")
#     f_prime = np.gradient(f_values, x)
#     log_deriv = f_prime / f_values
#     is_descending = np.all(np.diff(log_deriv) <= 0)
#     return is_descending

# def construct_envelope(hull_points, h, domain):
#     x_min, x_max = domain
#     h_values = h(hull_points)

#     slopes = np.gradient(h_values, hull_points)
#     intercepts = h_values - hull_points * slopes

#     print("DEBUG: Slopes:", slopes)
#     print("DEBUG: Intercepts:", intercepts)

#     z_points = [x_min]
#     for i in range(len(hull_points) - 1):
#         slope1, intercept1 = slopes[i], intercepts[i]
#         slope2, intercept2 = slopes[i + 1], intercepts[i + 1]

#         if np.abs(slope1 - slope2) < 1e-10:
#             raise ValueError("Consecutive slopes are too close; invalid envelope.")
#         z_intersect = (intercept2 - intercept1) / (slope1 - slope2)
#         z_points.append(z_intersect)

#     z_points.append(x_max)
#     pieces = [(float(slopes[i]), float(intercepts[i])) for i in range(len(slopes))]
#     print("DEBUG: Pieces (slopes, intercepts):", pieces)
#     print("DEBUG: z_points:", z_points)

#     return pieces, z_points

# def sample_piecewise_linear(pieces, z_points):
#     print("DEBUG: Sampling from envelope.")
#     areas, cumulative_areas = [], [0]
#     for i, (slope, intercept) in enumerate(pieces):
#         x_start, x_end = z_points[i], z_points[i + 1]
#         if slope == 0:
#             area = np.exp(intercept) * (x_end - x_start)
#         else:
#             exp_start = np.exp(intercept + slope * x_start)
#             exp_end = np.exp(intercept + slope * x_end)
#             area = (exp_end - exp_start) / slope
#         areas.append(area)
#         cumulative_areas.append(cumulative_areas[-1] + area)

#     total_area = cumulative_areas[-1]
#     u = np.random.uniform(0, total_area)
#     segment = np.searchsorted(cumulative_areas, u) - 1
#     slope, intercept = pieces[segment]
#     x_start, x_end = z_points[segment], z_points[segment + 1]

#     if slope == 0:
#         return np.random.uniform(x_start, x_end)
#     else:
#         cdf_start = np.exp(intercept + slope * x_start) / slope
#         cdf_sample = cdf_start + u - cumulative_areas[segment]
#         return (np.log(cdf_sample * slope) - intercept) / slope

# def ars(f, num_samples, x_init, domain=(-10, 10), burn_in=10):
#     """
#     Adaptive Rejection Sampling with debugging.

#     Args:
#         f (function): Target probability density function.
#         num_samples (int): Number of samples to generate.
#         x_init (list): Initial points for constructing envelope.
#         domain (tuple): Range of the distribution.
#         burn_in (int): Number of initial samples to discard.

#     Returns:
#         np.array: Array of sampled points.
#     """
#     print("DEBUG: Starting ARS.")
#     if not is_log_concave(f, np.linspace(*domain, 1000)):
#         raise ValueError("The input function is not log-concave!")

#     h = lambda x: h_log(f, x)
#     x_points = np.array(sorted(x_init))
#     samples = []
#     pieces, z_points = construct_envelope(x_points, h, domain)

#     for i in range(num_samples + burn_in):
#         print(f"DEBUG: Sampling from envelope. Iteration {i}")
        
#         # Sample from the envelope
#         x_star = sample_piecewise_linear(pieces, z_points)
#         u = np.random.uniform()
#         print(f"DEBUG: Iteration {i}, x_star={x_star}, u={u}")
        
#         # Check acceptance criteria
#         if u <= np.exp(h(x_star) - calculate_envelope(x_star, pieces, z_points)):
#             print(f"DEBUG: Accepted x_star={x_star}")
#             if i >= burn_in:  # Only append after burn-in
#                 samples.append(x_star)
#         else:
#             print(f"DEBUG: Rejected x_star={x_star}")
#             # Update the envelope with the new point
#             pieces, z_points = update_envelope(x_points, x_star, h, domain)
#             x_points = np.sort(np.append(x_points, x_star))

#     print(f"DEBUG: Finished sampling. Total samples collected: {len(samples)}")
#     return np.array(samples)

# def calculate_envelope(x, pieces, z_points):
#     for i in range(len(pieces)):
#         if z_points[i] <= x <= z_points[i+1]:
#             slope, intercept = pieces[i]
#             return intercept + slope * x

# def update_envelope(hull_points, new_point, h, domain):
#     hull_points = np.sort(np.append(hull_points, new_point))
#     return construct_envelope(hull_points, h, domain)

### V2 Below ###

import numpy as np
import scipy.stats as stats

def h_log(f, x):
    """Manual caching for log of the target function with underflow protection."""
    if isinstance(x, np.ndarray):
        x_key = tuple(x.tolist())  # Convert array to hashable tuple
    else:
        x_key = x

    if x_key not in h_log.cache:
        f_value = f(x)
        if isinstance(f_value, np.ndarray):
            f_value = np.maximum(f_value, np.finfo(float).eps)  # Prevent underflow for arrays
        else:
            f_value = max(f_value, np.finfo(float).eps)  # Prevent underflow for scalars
        h_log.cache[x_key] = np.log(f_value)

    return h_log.cache[x_key]

h_log.cache = {}

def is_log_concave(f, x_range, eps=1e-10):
    """Checks if a function is log-concave over the given range."""
    x = np.asarray(x_range)
    # Ensure no duplicates in range (for numerical differentiation)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x_range need to be increasing.")
    f_values = f(x)
    if np.any(f_values <= 0):
        raise ValueError("Function values must be positive.")
    f_values = np.maximum(f_values, eps)
    f_prime = np.gradient(f_values, x)
    log_deriv = f_prime / f_values
    is_descending = np.all(np.diff(log_deriv) <= 0)
    return is_descending

def construct_envelope(hull_points, h, domain):
    x_min, x_max = domain
    h_values = h(hull_points)

    slopes = np.gradient(h_values, hull_points)
    intercepts = h_values - hull_points * slopes

    z_points = [x_min]
    for i in range(len(hull_points) - 1):
        slope1, intercept1 = slopes[i], intercepts[i]
        slope2, intercept2 = slopes[i + 1], intercepts[i + 1]

        if np.abs(slope1 - slope2) < 1e-10:  # Avoid numerical instability
            raise ValueError("Consecutive slopes are too close; invalid envelope.")
        z_intersect = (intercept2 - intercept1) / (slope1 - slope2)
        z_points.append(z_intersect)

    z_points.append(x_max)
    pieces = [(float(slopes[i]), float(intercepts[i])) for i in range(len(slopes))]
    return pieces, z_points

def sample_piecewise_linear(pieces, z_points):
    areas, cumulative_areas = [], [0]
    for i, (slope, intercept) in enumerate(pieces):
        x_start, x_end = z_points[i], z_points[i + 1]
        if slope == 0:
            area = np.exp(intercept) * (x_end - x_start)
        else:
            exp_start = np.exp(np.clip(intercept + slope * x_start, -700, 700))  # Clip for overflow protection
            exp_end = np.exp(np.clip(intercept + slope * x_end, -700, 700))
            area = (exp_end - exp_start) / slope
        areas.append(area)
        cumulative_areas.append(cumulative_areas[-1] + area)

    total_area = cumulative_areas[-1]
    u = np.random.uniform(0, total_area)
    segment = np.searchsorted(cumulative_areas, u) - 1
    slope, intercept = pieces[segment]
    x_start, x_end = z_points[segment], z_points[segment + 1]

    if slope == 0:
        return np.random.uniform(x_start, x_end)
    else:
        cdf_start = np.exp(np.clip(intercept + slope * x_start, -700, 700)) / slope
        cdf_sample = cdf_start + u - cumulative_areas[segment]
        return (np.log(cdf_sample * slope) - intercept) / slope

def calculate_envelope(x, pieces, z_points):
    for i in range(len(pieces)):
        if z_points[i] <= x <= z_points[i + 1]:
            slope, intercept = pieces[i]
            return intercept + slope * x

def update_envelope(hull_points, new_point, h, domain):
    hull_points = np.sort(np.append(hull_points, new_point))
    return construct_envelope(hull_points, h, domain)

def initialize_points(f, domain, num_points=3):
    """Intelligently initialize starting points based on the target function."""
    x = np.linspace(domain[0], domain[1], num_points * 10)  # Use a dense grid for initialization
    f_values = f(x)
    sorted_indices = np.argsort(f_values)[::-1]  # Sort by PDF values, descending
    initial_points = x[sorted_indices[:num_points]]  # Choose the top `num_points` values
    print(f"DEBUG: Initializing points: {initial_points}")
    return initial_points

def ars(f, num_samples, domain=(-10, 10), burn_in=10, num_init_points=3):
    """
    Adaptive Rejection Sampling with intelligent initialization and overflow protection.

    Args:
        f (function): Target probability density function.
        num_samples (int): Number of samples to generate.
        domain (tuple): Range of the distribution.
        burn_in (int): Number of initial samples to discard.
        num_init_points (int): Number of initial points for constructing envelope.

    Returns:
        np.array: Array of sampled points.
    """
    print("DEBUG: Starting ARS.")
    if not is_log_concave(f, np.linspace(*domain, 1000)):
        raise ValueError("The input function is not log-concave!")

    h = lambda x: h_log(f, x)
    x_points = initialize_points(f, domain, num_points=num_init_points)
    samples = []
    pieces, z_points = construct_envelope(x_points, h, domain)

    for i in range(num_samples + burn_in):
        # Sample from the envelope
        x_star = sample_piecewise_linear(pieces, z_points)
        u = np.random.uniform()
        print(f"DEBUG: Iteration {i}, x_star={x_star}, u={u}")

        # Check acceptance criteria
        if u <= np.exp(h(x_star) - calculate_envelope(x_star, pieces, z_points)):
            print(f"DEBUG: Accepted x_star={x_star}")
            if i >= burn_in:  # Only append after burn-in
                samples.append(x_star)
        else:
            print(f"DEBUG: Rejected x_star={x_star}")
            # Update the envelope with the new point
            pieces, z_points = update_envelope(x_points, x_star, h, domain)
            x_points = np.sort(np.append(x_points, x_star))

    print(f"DEBUG: Finished sampling. Total samples collected: {len(samples)}")
    return np.array(samples)