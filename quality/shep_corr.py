import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


# Differently than the previous version, this one computes the correlation with the full
#   square matrix. I could not find a way to break the triangle matrix into chunks.
def shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high.flatten(), D_low.flatten())[0]


# TODO This only works if chunk_size=n
class ShepardCorrelation:

    def __partial(self, D_h, D_l):
        return stats.spearmanr(D_h.flatten(), D_l.flatten())[0]

    def __aggregate(self, partial_results, chunk_size):
        return np.mean(partial_results)
    
    def compute(self, X, P, chunk_size=None):
        n = X.shape[0]
        if not chunk_size:
            chunk_size = n
        partial_results = []
        for i in range(0, n, chunk_size):
            D_h = cdist(X[i:i+chunk_size], X)
            D_l = cdist(P[i:i+chunk_size], P)
            partial_results.append(self.__partial(D_h, D_l))
        return self.__aggregate(partial_results, chunk_size)


# TODO test these things instead with artificial clustered data, with random noise
# TODO refactor out a standard benchmark for all the metrics

if __name__ == "__main__":
    import random
    from time import perf_counter    
    
    n = 2500
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)

    t0 = perf_counter()
    sc_chunks = ShepardCorrelation().compute(X, P)
    print(f"sc_chunks = {sc_chunks}, time = {perf_counter() - t0}")
        
    from scipy.spatial.distance import squareform, pdist
    t0 = perf_counter()
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))
    sc_full = shepard_diagram_correlation(D_h, D_l)
    print(f"sc_full = {np.round(sc_full, 6)}, time = {perf_counter() - t0}")
    
    # Equal up to 8 decimals
    print("Same?", "Yes" if np.isclose(sc_chunks, sc_full) else "No")
