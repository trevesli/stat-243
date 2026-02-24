# ars-dev
Adaptive Rejection Sampling for STAT 243 Final Project at UC Berkeley, Fall 2024.

## Group Project (Originally developed in ars-dev)

This directory contains a copy of a team-based project.

My contributions include testing; installation and package setup; and finalization of the funtional structure

## Installation

```bash
pip install .
```

## Example Implementation

```python
import numpy as np
import ars

def gaussian(x):
    return np.exp(-0.5 * x**2)

if __name__ == "__main__":
    samples = ars.ars(gaussian, num_samples=10000, domain=(-3, 3))
```
