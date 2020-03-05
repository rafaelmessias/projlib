import numpy as np
from scipy.spatial.distance import cdist
from metric import DistanceBasedMetric


# FIXME: returning unbounded values. Check definitions on "Modern Multidimensional Scaling: Theory and Applications"
def normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low)**2) / np.sum(D_high**2) / 100

class NormalizedStress(DistanceBasedMetric):

    def partial(self, D_h, D_l):
        return np.sum((D_h - D_l)**2), np.sum(D_h**2)
    
    def aggregate(self, partial_results):
        num, den = [sum(x) for x in zip(*partial_results)]
        return num / den / 100


if __name__ == "__main__":
    import random
    from time import perf_counter
    
    n = random.randint(500, 1000)
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)

    t0 = perf_counter()
    ns_chunks = NormalizedStress().compute(X, P)
    print(f"ns_chunks = {ns_chunks}, time = {perf_counter() - t0}")
        
    from scipy.spatial.distance import squareform, pdist    
    t0 = perf_counter()
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))    
    ns_full = normalized_stress(D_h, D_l)
    print(f"ns_full = {ns_full}, time = {perf_counter() - t0}")

    # Equal up to 8 decimals
    print("Same?", "Yes" if np.isclose(ns_chunks, ns_full) else "No") 
