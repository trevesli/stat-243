# ars/__init__.py

# Import functions from ars.py so they are accessible at the package level
from .ars import (
    check_overflow_underflow,
    h_log,
    h_cached,
    is_log_concave,
    construct_envelope,
    sample_piecewise_linear,
    calculate_envelope,
    update_envelope,
    initialize_points,
    ars
)