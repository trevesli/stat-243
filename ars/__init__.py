# ars/__init__.py

# Define package interface
from .ars import ars

# Core sampling functions
from .sampler import ars, construct_envelope, sample_piecewise_linear, calculate_envelope, update_envelope

# Utility functions
from .utils import h_log

# Validation functions
from .validation import is_log_concave