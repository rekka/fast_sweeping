Python 3 module that exposes the `fast_sweeping` library.

## Installation

1. **Prerequisites.**

    * Make sure that [Rust is installed](https://www.rust-lang.org/en-US/install.html).

    * Install the Python package `setuptools-rust`:

        ```bash
        pip install setuptools-rust
        ```

2. Clone the repository and install the package:

    ```bash
    git clone https://github.com/rekka/fast_sweeping.git
    cd fast_sweeping/python
    python setup.py install
    ```

## Usage

```python
import numpy as np
import fast_sweeping

d = fast_sweeping.signed_distance(np.array([[-1., -1.], [1., 1.]]), 1.)
print(d)
```
