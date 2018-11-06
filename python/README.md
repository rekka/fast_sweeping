Python 3 module that exposes the `fast_sweeping` library.

## Installation

1. Make sure that [Rust is installed](https://www.rust-lang.org/en-US/install.html).

2. Install the Python package `setuptools-rust`:

    ```bash
    pip install setuptools-rust
    ```

3. Install the `fast_sweeping` package: In the `fast_sweeping/python`
   directory, run

    ```bash
    python setup.py install
    ```

## Usage

```python
import numpy as np
import fast_sweeping

d = fast_sweeping.signed_distance(np.array([[-1., -1.], [1., 1.]]), 1.)
print(d)
```
