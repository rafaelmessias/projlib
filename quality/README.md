# projlib.quality

Quality metrics for projections. 

## Available Metrics

* Trustworthiness
* Continuity
* NeighborhoodHit
* NormalizedStress
* ShepardCorrelation

For example, to compute the trustworthiness:

```python
import numpy as np
from projlib.quality.trustworthiness import Trustworthiness

X, P = np.random.rand(100, 10), np.random.rand(100, 2)
tw = Trustworthiness().compute(X, P, k=20)
print(tw)
```

Each metric class is in its own file and has a `__main__` with a simple test, which you can also check for an usage example.

## References

* https://ieeexplore.ieee.org/document/8851280/
* Code for quality metrics based on https://github.com/mespadoto/proj-quant-eval/
