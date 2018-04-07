Python 3 module that exposes the `fast_sweeping` library.

## Installation

```bash
python3 setup.py install
```

```python
import numpy as np
import fast_sweeping

d = fast_sweeping.signed_distance(np.array([[-1., -1.], [1., 1.]]), 1.)
print(d)
```
