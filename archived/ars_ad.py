import jax.numpy as jnp
from jax import grad, vmap
import jax.random as random


def is_log_concave(f, x_range):
    """
    Check if the function f is log-concave over the x_range.
    """
    x = jnp.asarray(x_range, dtype=jnp.float32)
    f_values = f(x)
    if jnp.any(f_values <= 0):
        raise ValueError("Function values must be positive.")

    log_f = lambda x: jnp.log(f(x))
    f_prime = vmap(grad(log_f))(x)
    log_deriv = f_prime
    return jnp.all(jnp.diff(log_deriv) <= 0)


def construct_envelope(hull_points, h, domain):
    """
    Construct the envelope function (upper bound) from hull points.
    """
    x_min, x_max = domain
    hull_points = jnp.asarray(hull_points, dtype=jnp.float32)
    h_values = h(hull_points)
    slopes = vmap(grad(h))(hull_points)
    intercepts = h_values - slopes * hull_points

    z_points = [x_min]
    for i in range(len(hull_points) - 1):
        slope1, intercept1 = slopes[i], intercepts[i]
        slope2, intercept2 = slopes[i + 1], intercepts[i + 1]
        z_intersect = (intercept2 - intercept1) / (slope1 - slope2)
        z_points.append(z_intersect)
    z_points.append(x_max)

    pieces = [(slopes[i], intercepts[i]) for i in range(len(slopes))]
    return pieces, jnp.array(z_points)


def construct_squeezing(hull_points, h, domain):
    """
    Construct the squeezing function (lower bound) from hull points.
    """
    x_min, x_max = domain
    hull_points = jnp.asarray(hull_points, dtype=jnp.float32)

    pieces = []
    z_points = [x_min]
    for i in range(len(hull_points) - 1):
        x1, x2 = hull_points[i], hull_points[i + 1]
        slope = (h(x2) - h(x1)) / (x2 - x1)
        intercept = h(x1) - slope * x1
        pieces.append((slope, intercept))
        z_points.append(x2)
    z_points.append(x_max)

    return pieces, jnp.array(z_points)

def calculate_squeezing(x, pieces, z_points):
    """
    Calculate squeezing function value.
    """
    for i in range(len(pieces)):
        if z_points[i] <= x <= z_points[i + 1]:
            slope, intercept = pieces[i]
            return intercept + slope * x


def calculate_envelope(x, pieces, z_points):
    """
    Calculate envelope function value.
    """
    for i in range(len(pieces)):
        if z_points[i] <= x <= z_points[i + 1]:
            slope, intercept = pieces[i]
            return intercept + slope * x

def sample_piecewise_linear(pieces, z_points, key):
    """
    Sample from the exponential of a piecewise linear function.
    """
    areas = []
    cumulative_areas = [0]
    for i, (slope, intercept) in enumerate(pieces):
        x_start, x_end = z_points[i], z_points[i + 1]
        if slope == 0:
            area = jnp.exp(intercept) * (x_end - x_start)
        else:
            area = (jnp.exp(intercept + slope * x_end) - jnp.exp(intercept + slope * x_start)) / slope
        areas.append(area)
        cumulative_areas.append(cumulative_areas[-1] + area)

    total_area = cumulative_areas[-1]
    u = random.uniform(key) * total_area
    segment = jnp.searchsorted(jnp.array(cumulative_areas), u) - 1

    slope, intercept = pieces[segment]
    x_start, x_end = z_points[segment], z_points[segment + 1]

    if slope == 0:
        x = random.uniform(key, minval=x_start, maxval=x_end)
    else:
        cdf_start = jnp.exp(intercept + slope * x_start) / slope
        cdf_sample = cdf_start + (u - cumulative_areas[segment])
        x = (jnp.log(cdf_sample * slope) - intercept) / slope

    return x


def ars(f, num_samples, x_init, domain=(-10.0, 10.0), key=random.PRNGKey(0)):
    """
    Adaptive Rejection Sampling using JAX.
    """
    if not is_log_concave(f, jnp.linspace(*domain, 1000, dtype=jnp.float32)):
        raise ValueError("The input function is not log-concave!")

    h = lambda x: jnp.log(f(x))
    x_points = jnp.array(sorted(x_init), dtype=jnp.float32)
    samples = []

    for _ in range(num_samples):
        envelope_pieces, envelope_points = construct_envelope(x_points, h, domain)
        squeezing_pieces, squeezing_points = construct_squeezing(x_points, h, domain)
        key, subkey = random.split(key)
        x_star = sample_piecewise_linear(envelope_pieces, envelope_points, subkey)

        u = random.uniform(key)
        if u <= jnp.exp(
            calculate_squeezing(x_star, squeezing_pieces, squeezing_points)
            - calculate_envelope(x_star, envelope_pieces, envelope_points)
        ):
            samples.append(x_star)
        elif u <= jnp.exp(h(x_star) - calculate_envelope(x_star, envelope_pieces, envelope_points)):
            samples.append(x_star)
            x_points = jnp.sort(jnp.append(x_points, x_star))
        else:
            x_points = jnp.sort(jnp.append(x_points, x_star))

    return jnp.array(samples)