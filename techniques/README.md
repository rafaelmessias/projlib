# projlib.techniques

One of the goals of projlib is to facilitate access to the vast amount of projection techniques (and variants) that exist out there. We will do that by aggregating many of them under a unified API, based on the well-known scikit-learn API. Right now only t-SNE (and a few variants) is available.

## Usage

```python
import numpy as np
from projlib.techniques.tsne import TSNE

X = np.random.rand(1000, 5)
p = TSNE(variant='fitsne').fit_transform(X)
print(p)

```