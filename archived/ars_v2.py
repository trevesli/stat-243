import numpy as np
import scipy.stats as stats

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
    
    f_prime = np.gradient(f_values, x)
    log_deriv = f_prime / f_values
    is_descending = np.all(np.diff(log_deriv) <= 0)
    
    return is_descending

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
    ## calculate the slopes and intercepts for each piece.
    slopes = np.gradient(h_values, hull_points)
    intercepts = h_values - hull_points * slopes
    
    ## calculate the start and end points for each piece.
    z_points = [x_min]
    for i in range(len(hull_points)-1):
        slope1, intercept1 = slopes[i], intercepts[i]
        slope2, intercept2 = slopes[i + 1], intercepts[i + 1]
        
        if np.abs(slope1 - slope2) < 1e-10:  # Avoid numerical instability
            raise ValueError("Consecutive slopes are too close; hull points must form a valid envelope.")
        z_intersect = (intercept2 - intercept1) / (slope1 - slope2)
        z_points.append(z_intersect)
    z_points.append(x_max)
    
    pieces = [(slopes[i], intercepts[i]) for i in range(len(slopes))]
    
    return pieces, z_points

def calculate_envelope(x, pieces, z_points):
    """
    Calculate the value on x for the envelope function.
    
    Args:
        x(float): the point we want to calculate.
        pieces(list of tuples): List of (slope, intercept) for each segment.
        z_points(array_like): Start and end points of each segment.
    
    Returns:
        float: value of the envelope function at x.
    """
    
    for i in range(len(pieces)):
        if x >= z_points[i] and x <= z_points[i+1]:
            slope, intercept = pieces[i]
            return intercept + slope * x

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

def calculate_squeezing(x, pieces, z_points):
    """
    Calculate the value on x for the squeezing function.
    
    Args:
        x(float): the point we want to calculate.
        pieces(list of tuples): List of (slope, intercept) for each segment.
        z_points(array_like): Start and end points of each segment.
    
    Returns:
        float: value of the squeezing function at x.
    """
    
    for i in range(len(pieces)):
        if x >= z_points[i] and x <= z_points[i+1]:
            slope, intercept = pieces[i]
            return intercept + slope * x

def sample_piecewise_linear(pieces, z_points):
    """
    Sample from the exponential of a piecewise linear function.
    
    Args:
        pieces: List of tuples (slope, intercept) representing line segments.
        z_points: List of the start and end of each piece.
    
    Returns:
        A sampled point from the exponential of piecewise linear function.
    """
    # Calculate cumulative areas under each segment
    areas = []
    cumulative_areas = [0]  # Start with 0 for cumulative sum
    for i, (slope, intercept) in enumerate(pieces):
        x_start, x_end = z_points[i], z_points[i + 1]
        if slope == 0: 
            area = np.exp(intercept) * (x_end - x_start)
        else:
            # Integral of exp(intercept + slope * x) from x_start to x_end
            area = (np.exp(intercept + slope * x_end) - np.exp(intercept + slope * x_start)) / slope
        areas.append(area)
        cumulative_areas.append(cumulative_areas[-1] + area)

    total_area = cumulative_areas[-1]  # Total area under the piecewise function
    
    # Sample a uniform random value in [0, total_area] to pick a segment
    u = np.random.uniform(0, total_area)
    
    # Find the segment corresponding to the sampled area
    segment = np.searchsorted(cumulative_areas, u) - 1  # Adjust index because cumulative_areas starts with 0
    slope, intercept = pieces[segment]
    x_start, x_end = z_points[segment], z_points[segment + 1]
    
    # Sample within the chosen segment
    if slope == 0:
        # Uniform sampling for horizontal segments
        x = np.random.uniform(x_start, x_end)
    else:
        # Solve for x in the CDF of the exponential of a line
        cdf_start = np.exp(intercept + slope * x_start) / slope
        cdf_sample = cdf_start + u - cumulative_areas[segment]
        x = (np.log(cdf_sample * slope) - intercept) / slope
    
    return x

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

def ars_init_domain(f, num_samples, x_init, domain=(-10,10), burn_in = 10000):
    """
    Adaptive Rejection Sampling given domain and initial points.
    
    Args:
        f(function): Probability density function (maybe unnormalized)
        num_samples(int): Number of samples to generate
        x_init(array_like): Initial points to start the algorithm
        domain(tuple): (x_min, x_max), range of the distribution
        
    Returns:
        samples(array_like): num_samples number of samples from f.
    """
    
    if not is_log_concave(f, np.linspace(*domain, 1000)):
        raise ValueError("The input function is not log-concave!")

    h = lambda x: np.log(f(x))
    x_points = np.array(sorted(x_init))
    samples = []

    while len(samples)<= (num_samples + burn_in):
        # Calculate tangents and chords
        envelope_pieces, envelope_points = construct_envelope(x_points, h, domain)
        squeezing_pieces, squeezing_points = construct_squeezing(x_points, h, domain)

        # Sample from envelope
        x_star = sample_piecewise_linear(envelope_pieces, envelope_points)

        # Accept/reject
        u = np.random.uniform()
        if u <= np.exp(calculate_squeezing(x_star, squeezing_pieces, squeezing_points) - calculate_envelope(x_star, envelope_pieces, envelope_points)):  # Check lower bound
            samples.append(x_star)
        elif u <= np.exp(h(x_star) - calculate_envelope(x_star, envelope_pieces, envelope_points)):  # Accept
            samples.append(x_star)
            x_points = np.sort(np.append(x_points, x_star))
        else:
            x_points = np.sort(np.append(x_points, x_star))
            
    samples = samples[burn_in+1:]
    return np.array(samples)

def ars(f, num_samples):
    """
    Adaptive Rejection Sampling.
    
    Args:
        f(function): Probability density function (maybe unnormalized)
        num_samples(int): Number of samples to generate
    
    Returns:
        samples(array_like): num_samples number of samples from f.
    """
    domain = adaptive_search_domain(f)
    init_1, init_2 = init_points(f, domain)
    x_init = np.linspace(init_1, init_2, 10)
    samples = ars_init_domain(f, num_samples, x_init = x_init, domain = domain)
    return samples 


