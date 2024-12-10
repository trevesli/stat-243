# ars-dev
Adaptive Rejection Sampling for STAT 243 Final Project at UC Berkeley, Fall 2024.

## Installation

```bash
pip install .
```

## Example Implementation

```python
import numpy as np
import ars

def target_function(x):
    return np.exp(-0.5 * (x - 2) ** 2)

if __name__ == "__main__":
    samples = ars.ars(target_function, num_samples=1000, domain=(-3, 3))
```