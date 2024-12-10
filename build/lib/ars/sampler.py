import numpy as np
from .utils import check_overflow_underflow, h_log, h_cached
from .validation import is_log_concave

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
    # Input checks
    if len(z_points) != len(pieces) + 1:
        raise ValueError(f"Length of `z_points` ({len(z_points)}) should be length of `pieces` ({len(pieces)}) + 1")

    areas, cumulative_areas = [], [0]
    
    for i, (slope, intercept) in enumerate(pieces):
        x_start, x_end = z_points[i], z_points[i + 1]
        if slope == 0:
            area = np.exp(intercept) * (x_end - x_start)
        else:
            # Overflow protection
            log_start = intercept + slope * x_start
            log_end = intercept + slope * x_end
            log_start, log_end = check_overflow_underflow([log_start, log_end])
            area = (np.exp(log_end) - np.exp(log_start)) / slope
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


"""Superseded with version below, that updates envelope incrementally
def update_envelope(hull_points, new_point, h, domain):
    hull_points = np.sort(np.append(hull_points, new_point))
    return construct_envelope(hull_points, h, domain)
"""

def update_envelope(x_points, new_point, h, slopes, intercepts, z_points, domain):
    # Insert the new point into the sorted list of x_points
    x_points = np.sort(np.append(x_points, new_point))

    # Update slopes and intercepts based on the new point
    slopes = []
    intercepts = []
    for idx in range(1, len(x_points)):
        slopes.append((h(x_points[idx]) - h(x_points[idx - 1])) / (x_points[idx] - x_points[idx - 1]))
        intercepts.append(h(x_points[idx]) - slopes[-1] * x_points[idx])

    # Update z_points to reflect the new x_points and corresponding function values
    z_points = np.array([h(x) for x in x_points])  # This ensures z_points has length len(x_points)

    # Ensure the lengths of pieces and z_points are aligned
    pieces = list(zip(slopes, intercepts))

    # Print debug info for checking
    print(f"DEBUG: Updated Envelope - z_points length: {len(z_points)}, pieces length: {len(pieces)}")

    return x_points, slopes, intercepts, z_points
    
def adaptive_search_domain(f, start=0, step=1, threshold=1e-15, max_steps=int(1e7)):
    """
    Search the domain of the given function.
    
    Args:
    - f(function): Given function we want to search for domain of. 
    - start(float): Starting searching point.
    - step(float): Searching step.
    - threshold(float): Threshold to judge whether the function value is too small
    - max_steps(int): maximum searching steps.
    
    Returns:
    - domain_start, domain_end(float): The searching domain of the function.
    """
    
    x = start
    domain_points = []
    
    # Searching towards the positive side
    for _ in range(max_steps):
        if f(x) > threshold:
            domain_points.append(x)
        x += step
    
    x = start
    
    # Searching towards the negative side
    for _ in range(max_steps):
        if f(x) > threshold:
            domain_points.append(x)
        else:
            break
        x -= step
    
    if not domain_points:
        return None
    return min(domain_points), max(domain_points)

def init_points(f, domain, threshold = 1e-5):
    """
    Search the initial point of the .
    
    Args:
    - f(function): Given function. 
    - domain(tuple): domain of the function.
    - threshold(float): Threshold to judge whether to be the initial point.
    
    Returns:
    - init_1, init_2(tuple): Two initial points.
    """
    
    domain_min, domain_max = adaptive_search_domain(f)
    step = (domain_max - domain_min) / 1000
    
    init_1 = domain_min
    while f(init_1) <= threshold:
        init_1 += step
    
    init_2 = domain_max
    while f(init_2) <= threshold:
        init_2 -= step
        
    return init_1, init_2

def ars(f, num_samples, domain, burn_in=1000, num_init_points=10):
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
    print("Starting ARS ...")
    print("Checking if the function is log-concave ...")
    if not is_log_concave(f, np.linspace(*domain, 1000)):
        raise ValueError("The input function is not log-concave!")

    h = lambda x: h_log(f, x)
    
    domain = adaptive_search_domain(f)
    init_1, init_2 = init_points(f, domain)
    x_points = np.linspace(init_1, init_2, num_init_points)
    
    samples = []
    pieces, z_points = construct_envelope(x_points, h, domain)
    slopes, intercepts = zip(*pieces)

    for i in range(num_samples + burn_in):
        # Sample from the envelope
        x_star = sample_piecewise_linear(pieces, z_points)
        u = np.random.uniform()
        print(f"DEBUG: Iteration {i}, x_star={x_star}, u={u}")

        # Check acceptance criteria
        if u <= np.exp(h_cached(f, x_star) - calculate_envelope(x_star, pieces, z_points)):
            print(f"DEBUG: Accepted x_star={x_star}")
            if i >= burn_in:  # Only append after burn-in
                samples.append(x_star)
        else:
            print(f"DEBUG: Rejected x_star={x_star}")
            slopes = list(slopes)  # Convert slopes to list (if tuple)
            intercepts = list(intercepts)  # Convert intercepts to list (if tuple)
            z_points = list(z_points)

            # Incrementally update the envelope with the new point
            x_points, slopes, intercepts, z_points = update_envelope(
                x_points, x_star, h, slopes, intercepts, z_points, domain
            )
            pieces = list(zip(slopes, intercepts))

    print(f"Finished sampling. Total samples collected: {len(samples)}")
    return np.array(samples)