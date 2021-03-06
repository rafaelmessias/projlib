import numpy as np
from scipy.spatial.distance import cdist
from projlib.quality.metric import DistanceBasedMetric
from numba import jit
from setdiff1d import setdiff1d
from sklearn.neighbors import NearestNeighbors


def trustworthiness(D_high, D_low, k=7):

    n = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0][0] - k

        sum_i += sum_j

    return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)


# TODO The numba version is performing bad right now. Maybe it's because sorting in numba is slower than in numpy, and it also does not support arrays with more than 1D. Also my implementation of setdiff1d is not very good.

@jit(nopython=True, parallel=True)
def numba_tw(D_high, D_low, k=7):
    n = D_high.shape[0]
    
    nn_orig, nn_proj = np.empty_like(D_high), np.empty_like(D_low)
    for i in range(n):
        nn_orig[i] = D_high[i].argsort()
        nn_proj[i] = D_low[i].argsort()

    knn_orig = nn_orig[:, 1:k + 1]
    knn_proj = nn_proj[:, 1:k + 1]

    sum_i = 0

    for i in range(n):
        #U = np.setdiff1d(knn_proj[i], knn_orig[i])
        U = setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            v = np.where(nn_orig[i] == U[j])[0][0]            
            sum_j += v - k

        sum_i += sum_j

    return 1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)


# TODO  Make importing a bit easier; right now it's:
#           "from projlib.quality.trustworthiness import Trustworthiness",
#       which is a bit too long (and duplicated).

class Trustworthiness(DistanceBasedMetric):

    def partial(self, D_h, D_l, k):        
        n = D_h.shape[0]

        nn_orig = D_h.argsort()
        nn_proj = D_l.argsort()

        knn_orig = nn_orig[:, :k + 1][:, 1:]
        knn_proj = nn_proj[:, :k + 1][:, 1:]

        sum_i = 0

        for i in range(n):
            U = np.setdiff1d(knn_proj[i], knn_orig[i])

            sum_j = 0
            for j in range(U.shape[0]):
                # This is so stupid...
                sum_j += np.where(nn_orig[i] == U[j])[0] - k

            sum_i += sum_j

        return sum_i

    def aggregate(self, partial_results, k):
        n = self.X.shape[0]
        total = sum(partial_results)
        return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * total)).squeeze())


if __name__ == "__main__":
    import random
    from time import perf_counter
    from scipy.spatial.distance import squareform, pdist
    
    n = 5000
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)
    k = int(np.sqrt(n))
    
    t0 = perf_counter()
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))
    tw_full = trustworthiness(D_h, D_l, k=k)
    print(f"tw_full = {tw_full}, time = {perf_counter() - t0}")

    # Compile
    # numba_tw(D_h, D_l, k=k)
    # t0 = perf_counter()
    # tw_numba = numba_tw(D_h, D_l, k=k)
    # print(f"tw_numba = {tw_full}, time = {perf_counter() - t0}")
    # print("Same?", "Yes" if np.isclose(tw_full, tw_numba) else "No")

    t0 = perf_counter()
    tw_chunks = Trustworthiness().compute(X, P, k=k)
    print(f"tw_chunks = {tw_chunks}, time = {perf_counter() - t0}")
    # Equal up to 8 decimals
    print("Same?", "Yes" if np.isclose(tw_chunks, tw_full) else "No")

    t0 = perf_counter()
    tw_chunks_jobs = Trustworthiness().compute(X, P, k=k, n_jobs=-1)
    print(f"tw_chunks_jobs = {tw_chunks_jobs}, time = {perf_counter() - t0}")
    print("Same?", "Yes" if np.isclose(tw_chunks_jobs, tw_chunks) else "No")
