import numpy as np
from .utils import h_log
from .validation import is_log_concave

def construct_envelope(hull_points, h, domain):
    """
    Construct the envelope function (upper bound) from hull points.
    
    Args:
        hull_points(array-like): x-values where tangents are constructed.
        h: the log of sampling function.
        domain(tuple): (x_min, x_max), the domain boundaries of the distribution.
    
    Returns:
        pieces(list of tuples): List of (slope, intercept) for each segment.
        z_points(array_like): Start and end points of each segment.
    """
    
    if len(hull_points) < 3:
        raise ValueError("There must be at least 3 hull points.")
    
    x_min, x_max = domain
    h_values = h(hull_points)

    slopes = [(h(x+1e-5)-h(x-1e-5))/2e-5 for x in hull_points]
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

def construct_squeezing(hull_points, h, domain):
    """
    Construct the squeezing function (lower bound) from hull points.
    
    Args:
        hull_points(array-like): x-values where chords are constructed.
        h: the log of sampling function.
        domain(tuple): (x_min, x_max), the domain boundaries of the distribution.
    
    Returns:
        pieces(list of tuples): List of (slope, intercept) for each segment.
        z_points(array_like): Start and end points of each segment.
    """
    if len(hull_points) < 3:
        raise ValueError("There must be at least 3 hull points.")
    
    x_min, x_max = domain
    pieces = []
    z_points = [x_min, hull_points[0]]
    
    if x_min == -np.inf:
        pieces.append((0, -np.inf))
    else:
        x1, x2 = x_min, hull_points[0]
        slope = (h(x2)-h(x1))/ (x2 - x1)
        intercept = h(x1) - slope * x1
        pieces.append((slope, intercept))
    
    for i in range(len(hull_points)-1):
        x1, x2 = hull_points[i], hull_points[i+1]
        slope = (h(x2)-h(x1))/ (x2 - x1)
        intercept = h(x1) - slope * x1
        pieces.append((slope, intercept))
        z_points.append(x2)
    
    if x_max == np.inf:
        pieces.append((0, -np.inf))
    else:
        x1, x2 = hull_points[-1], x_max
        slope = (h(x2)-h(x1))/ (x2 - x1)
        intercept = h(x1) - slope * x1
        pieces.append((slope, intercept))
        
    z_points.append(x_max)
    
    return pieces, z_points

def calculate_piecewise_linear(x, pieces, z_points):
    """
    Calculate the value on x for a piecewise linear function.
    
    Args:
        x(float): the point we want to calculate.
        pieces(list of tuples): List of (slope, intercept) for each segment.
        z_points(array_like): Start and end points of each segment.
    
    Returns:
        (float): value of the squeezing function at x.
    """
    
    for i in range(len(pieces)):
        if x > z_points[i] and x <= z_points[i + 1]:
            slope, intercept = pieces[i]
            return intercept + slope * x

def update_envelope(h, orig_x_points, orig_pieces, orig_z_points, new_point):
    """
    Add a new point to the update the envelope function.
    
    Args:
        h(function): the log of sampling function f.
        orig_x_points(List): List of the original hull points.
        orig_pieces(list): List of tuples (slope, intercept) representing line segments.
        orig_z_points(list): List of the start and end of each piece.
        new_point(float): new point need to be added to be a new hull point.
        
    Returns:
        new_pieces(list of tuples): List of (slope, intercept) for each segment.
        new_z_points(array_like): Start and end points of each segment.
    """
    domain_min, domain_max = orig_z_points[0], orig_z_points[-1]
    if new_point <= domain_min or new_point >= domain_max:
        raise ValueError(f"New point is out of the domain of the function.")
    
    new_slope = (h(new_point+1e-5) - h(new_point-1e-5))/2e-5
    new_intercept = h(new_point) - new_point * new_slope
    if domain_min < new_point <= orig_x_points[0]:
        if new_point == orig_x_points[0]:
            print("New points already in the hull points.")
            return orig_pieces, orig_z_points
        ## calculate new z point
        slope, intercept = orig_pieces[0]
        z =  -(intercept - new_intercept) / (slope - new_slope)
        orig_z_points.insert(1, z)
        ## insert new piece
        orig_pieces.insert(0, (new_slope, new_intercept))
        return orig_pieces, orig_z_points
    
    for i in range(len(orig_x_points)-1):
        if orig_x_points[i] < new_point <= orig_x_points[i+1]:
            if new_point == orig_x_points[i+1]:
                print("New points already in the hull points.")
                return orig_pieces, orig_z_points

            ## calculate new z_points
            slope1, intercept1 = orig_pieces[i]
            slope2, intercept2 = orig_pieces[i+1]
            z1 = -(intercept1 - new_intercept) / (slope1 - new_slope)
            z2 = -(intercept2 - new_intercept) / (slope2 - new_slope)
            orig_z_points[i+1] = z2
            orig_z_points.insert(i+1, z1)
            ## insert new piece
            orig_pieces.insert(i+1, (new_slope, new_intercept))
            return orig_pieces, orig_z_points
    
    if orig_x_points[-1] < new_point < domain_max:
        ## calculate new z point
        slope, intercept = orig_pieces[0]
        z = -(intercept - new_intercept) / (slope - new_slope)
        orig_z_points.insert(-1, z)
        ## insert new piece
        orig_pieces.append((new_slope, new_intercept))
        return orig_pieces, orig_z_points

def update_squeezing(h, orig_pieces, orig_z_points, new_point):
    """
    Add a new point to the update the squeezing function.
    
    Args:
        h(function): the log of sampling function f.
        orig_pieces(list): List of tuples (slope, intercept) representing line segments.
        orig_z_points(list): List of the start and end of each piece.
        new_point(float): new point need to be added to be a new hull point.
        
    Returns:
        new_pieces(list of tuples): List of (slope, intercept) for each segment.
        new_z_points(array_like): Start and end points of each segment.
    """
    domain_min, domain_max = orig_z_points[0], orig_z_points[-1]
    if new_point <= domain_min or new_point >= domain_max:
        raise ValueError(f"New point is out of the domain of the function.")
    
    for i in range(len(orig_z_points)-1):
        if orig_z_points[i] < new_point <= orig_z_points[i+1]:
            if new_point == orig_z_points[i+1]:
                print("New points already in the hull points.")
                return orig_pieces, orig_z_points
            ## insert new piece
            slope1 = (h(new_point) - h(orig_z_points[i]))/(new_point - orig_z_points[i])
            intercept1 = h(new_point) - slope1 * new_point
            slope2 = (h(new_point) - h(orig_z_points[i+1]))/(new_point - orig_z_points[i+1])
            intercept2 = h(new_point) - slope2 * new_point
            orig_pieces[i] = (slope1, intercept1)
            orig_pieces.insert(i+1, (slope2, intercept2))
            ## insert new z point
            orig_z_points.insert(i+1, new_point)
            return orig_pieces, orig_z_points

def sample_piecewise_linear(pieces, z_points):
    """
    Sample from the exponential of a piecewise linear function.
    
    Args:
        pieces: List of tuples (slope, intercept) representing line segments.
        z_points: List of the start and end of each piece.
    
    Returns:
        A sampled point from the exponential of piecewise linear function.
    """
    # Input checks
    if len(z_points) != len(pieces) + 1:
        raise ValueError(f"Length of `z_points` ({len(z_points)}) should be length of `pieces` ({len(pieces)}) + 1")
    for i in range(len(z_points)-1):
        if z_points[i] >= z_points[i+1]:
            raise ValueError(f"Something wrong with the sampling procedure, please check the log-concaveness of the function.")

    areas, cumulative_areas = [], [0]
    
    for i, (slope, intercept) in enumerate(pieces):
        x_start, x_end = z_points[i], z_points[i + 1]
        if slope == 0:
            area = np.exp(intercept) * (x_end - x_start)
        else:
            log_start = intercept + slope * x_start
            log_end = intercept + slope * x_end
            area = (np.exp(log_end) - np.exp(log_start)) / slope
        areas.append(area)
        cumulative_areas.append(cumulative_areas[-1] + area)

    total_area = cumulative_areas[-1]
    
    # Sample a uniform random value in [0, total_area] to pick a segment
    u = np.random.uniform(0, total_area)
    # Find the segment corresponding to the sampled area
    segment = np.searchsorted(cumulative_areas, u) - 1
    slope, intercept = pieces[segment]
    x_start, x_end = z_points[segment], z_points[segment + 1]

    if slope == 0:
        return np.random.uniform(x_start, x_end)
    else:
        cdf_start = np.exp(np.clip(intercept + slope * x_start, -700, 700)) / slope
        cdf_sample = cdf_start + u - cumulative_areas[segment]
        return (np.log(cdf_sample * slope) - intercept) / slope
    
def adaptive_search_domain(f, start=0, step=0.1, threshold=1e-15, max_steps=int(1e6)):
    """
    Search the domain of the given function.
    
    Args:
    - f(function): Given function we want to search for domain of. 
    - start(float): Starting searching point.
    - step(float): Searching step.
    - threshold(float): Threshold to judge whether the function value is too small.
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
        raise ValueError("Cannot find the domain in the region, please check your function or change the argument                                 threshold, step or max_steps.")
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
    
    domain_min, domain_max = domain
    step = (domain_max - domain_min) / 1000
    
    init_1 = domain_min
    while f(init_1) <= threshold:
        init_1 += step
    
    init_2 = domain_max
    while f(init_2) <= threshold:
        init_2 -= step
        
    return init_1, init_2 


def ars(f, num_samples, domain=(-np.inf, np.inf), domain_threshold=1e-15, domain_step=0.1, max_step=int(1e6), burn_in=1000, init_threshold=1e-5, num_init_points=10):
    """
    Adaptive Rejection Sampling with intelligent initialization and overflow protection.

    Args:
        f (function): Target probability density function.
        num_samples (int): Number of samples to generate.
        domain (tuple): Range of the distribution.
        domian_threshold(float): Threshold to judge whether the function value is too small.
        domain_step(float): step size in adaptive domain search.
        max_step(int): max step in adaptive domain search
        burn_in (int): Number of initial samples to discard.
        init_threshold(float): threshold to find the initial points
        num_init_points (int): Number of initial points for constructing envelope( must be >=3).

    Returns:
        np.array: Array of sampled points.
    """

    # Input checks
    if not isinstance(domain, tuple):
        raise TypeError("'domain' must be a tuple.")
    if len(domain) != 2:
        raise ValueError("'domain' must contain exactly two elements.")
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not (isinstance(domain[0], (int, float)) and isinstance(domain[1], (int, float))):
        raise ValueError("Invalid domain: values must be numeric.")
    if not (isinstance(domain_threshold, (int, float))) or not (domain_threshold > 0):
        raise ValueError("'domain_threshold' must be a positive number.")
    if not (isinstance(domain_step, (int, float))) or not (domain_step > 0):
        raise ValueError("'domain_step' must be a positive number.")
    if not (isinstance(init_threshold, (int, float))) or not (init_threshold > 0):
        raise ValueError("'init_threshold' must be a positive number.")
    if not isinstance(max_step, int) or max_step <= 0:
        raise ValueError("'max_step' must be a positive integer.")
    if domain[0] >= domain[1]:
        raise ValueError("Lower bound of 'domain' must be less than upper bound.")
    if not callable(f):
        raise TypeError("Must use a callable function 'f'.")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("'burn_in' must be a non-negative integer.")
    if not isinstance(num_init_points, int) or num_init_points < 3:
        raise ValueError("'num_init_points' must be an integer >= 3.")
    
    print("Starting ARS ...")
    print("Searching for the domain ...")
    if domain == (-np.inf, np.inf):
        domain = adaptive_search_domain(f, threshold = domain_threshold)

    h = lambda x: h_log(f, x)
    
    init_1, init_2 = init_points(f, domain, threshold = init_threshold)
    x_points = np.linspace(init_1, init_2, num_init_points)
    samples = []
    envelope_pieces, envelope_points = construct_envelope(x_points, h, domain)
    squeezing_pieces, squeezing_points = construct_squeezing(x_points, h, domain)
    
    ## Check the log-concaveness of the function
    x_check = np.linspace(envelope_points[0]+1e-10, envelope_points[-1]-1e-10, 1000)
    for x in x_check:
        if calculate_piecewise_linear(x, squeezing_pieces, squeezing_points) > calculate_piecewise_linear(x, envelope_pieces, envelope_points):
            raise ValueError("The input function is not log-concave!")

    while len(samples) <= (num_samples + burn_in-1):
        # Sample from the envelope
        x_star = sample_piecewise_linear(envelope_pieces, envelope_points)
        u = np.random.uniform()
        
        if calculate_piecewise_linear(x_star, squeezing_pieces, squeezing_points) > calculate_piecewise_linear(x_star, envelope_pieces, envelope_points):
            raise ValueError("The input function is not log-concave!")

        # Check acceptance criteria
        if u <= np.exp(calculate_piecewise_linear(x_star, squeezing_pieces, squeezing_points) -               calculate_piecewise_linear(x_star, envelope_pieces, envelope_points)):  # Check lower bound
            samples.append(x_star)
            
        elif u <= np.exp(h(x_star) - calculate_piecewise_linear(x_star, envelope_pieces, envelope_points)):
            samples.append(x_star)
            envelope_pieces, envelope_points = update_envelope(h, x_points, envelope_pieces, envelope_points, x_star)
            squeezing_pieces, squeezing_points = update_squeezing(h, squeezing_pieces, squeezing_points, x_star)
            x_points = np.sort(np.append(x_points, x_star))
        else:
            envelope_pieces, envelope_points = update_envelope(h, x_points, envelope_pieces, envelope_points, x_star)
            squeezing_pieces, squeezing_points = update_squeezing(h, squeezing_pieces, squeezing_points, x_star)
            x_points = np.sort(np.append(x_points, x_star))

    samples = np.array(samples[burn_in:])

    # Output checks
    if not isinstance(samples, np.ndarray):
        raise TypeError("The output 'samples' must be a numpy array.")
    if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
        raise ValueError("Samples contain NaN or infinite values.")

    print(f"Finished sampling. Total samples collected: {len(samples)}")
    
    return samples

##### REVISED ARS WITH BATCHING #####
# def ars(f, num_samples, domain=(-np.inf, np.inf), domain_threshold=1e-15, domain_step=0.1, max_step=int(1e6), 
#         init_threshold=1e-5, num_init_points=10, batch_size=100):
#     """
#     Adaptive Rejection Sampling with batch processing and efficient updates.

#     Args:
#         f (function): Target probability density function.
#         num_samples (int): Number of samples to generate.
#         domain (tuple): Range of the distribution.
#         domain_threshold (float): Threshold to determine if function value is too small.
#         domain_step (float): Step size in adaptive domain search.
#         max_step (int): Max step in adaptive domain search.
#         init_threshold (float): Threshold to find the initial points.
#         num_init_points (int): Number of initial points for constructing envelope (must be >=3).
#         batch_size (int): Number of samples to generate per batch before updating the envelope.

#     Returns:
#         np.array: Array of sampled points.
#     """
#     # Input checks
#     if len(domain) != 2:
#         raise ValueError("'domain' must contain exactly two elements.")
#     if num_samples <= 0:
#         raise ValueError("num_samples must be a positive integer.")
#     if batch_size <= 0:
#         raise ValueError("batch_size must be a positive integer.")
    
#     print("Starting ARS...")
#     print("Searching for the domain...")
#     if domain == (-np.inf, np.inf):
#         mode_value = mode(np.linspace(-5, 5, 100), axis=None).mode[0]  # Estimate the mode heuristically
#         domain = adaptive_search_domain(f, start=mode_value, threshold=domain_threshold)
    
#     print("Checking if the function is log-concave...")
#     if not is_log_concave(f, np.linspace(*domain, 1000)):
#         raise ValueError("The input function is not log-concave!")

#     h = lambda x: h_log(f, x)
#     init_1, init_2 = init_points(f, domain, threshold=init_threshold)
#     x_points = np.linspace(init_1, init_2, num_init_points)
#     samples = []
    
#     # Initialize envelope and squeezing functions
#     envelope_pieces, envelope_points = construct_envelope(x_points, h, domain)
#     squeezing_pieces, squeezing_points = construct_squeezing(x_points, h, domain)

#     while len(samples) < num_samples:
#         batch_samples = []
        
#         for _ in range(batch_size):
#             # Sample from the envelope
#             x_star = sample_piecewise_linear(envelope_pieces, envelope_points)
#             u = np.random.uniform()
            
#             # Check acceptance criteria
#             if u <= np.exp(calculate_piecewise_linear(x_star, squeezing_pieces, squeezing_points) - 
#                            calculate_piecewise_linear(x_star, envelope_pieces, envelope_points)):
#                 batch_samples.append(x_star)
#             elif u <= np.exp(h(x_star) - calculate_piecewise_linear(x_star, envelope_pieces, envelope_points)):
#                 batch_samples.append(x_star)
#                 x_points = np.sort(np.append(x_points, x_star))

#         # Update envelope and squeezing functions with the batch
#         for x_star in batch_samples:
#             envelope_pieces, envelope_points = update_envelope(h, x_points, envelope_pieces, envelope_points, x_star)
#             squeezing_pieces, squeezing_points = update_squeezing(h, squeezing_pieces, squeezing_points, x_star)
        
#         samples.extend(batch_samples[:num_samples - len(samples)])

#     samples = np.array(samples)

#     # Output checks
#     if not isinstance(samples, np.ndarray):
#         raise TypeError("The output 'samples' must be a numpy array.")
#     if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
#         raise ValueError("Samples contain NaN or infinite values.")

#     print(f"Finished sampling. Total samples collected: {len(samples)}")
    
#     return samples

