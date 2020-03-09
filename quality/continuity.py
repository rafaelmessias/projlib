import numpy as np
from scipy.spatial.distance import cdist
from projlib.quality.metric import DistanceBasedMetric


def continuity(D_high, D_low, k=7):
    
    n = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


class Continuity(DistanceBasedMetric):
    
    def partial(self, D_high, D_low, k):
        n = D_high.shape[0]

        nn_orig = D_high.argsort()
        nn_proj = D_low.argsort()

        knn_orig = nn_orig[:, :k + 1][:, 1:]
        knn_proj = nn_proj[:, :k + 1][:, 1:]

        sum_i = 0

        for i in range(n):
            V = np.setdiff1d(knn_orig[i], knn_proj[i])

            sum_j = 0
            for j in range(V.shape[0]):
                sum_j += np.where(nn_proj[i] == V[j])[0] - k

            sum_i += sum_j
        
        return sum_i

    def aggregate(self, partial_results, k):
        n = self.X.shape[0]
        total = sum(partial_results)
        return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * total)).squeeze())


if __name__ == "__main__":
    import random
    from time import perf_counter
    
    n = 5000
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)

    t0 = perf_counter()
    co_chunks_jobs = Continuity().compute(X, P, k=20, chunk_size=10, chunk_search=False, n_jobs=-1)
    print(f"co_chunks_jobs = {co_chunks_jobs}, time = {perf_counter() - t0}")

    t0 = perf_counter()
    co_chunks = Continuity().compute(X, P, k=20)
    print(f"co_chunks = {co_chunks}, time = {perf_counter() - t0}")

    print("Same?", "Yes" if co_chunks_jobs == co_chunks else "No")

    from scipy.spatial.distance import squareform, pdist
    t0 = perf_counter()
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))    
    co_full = continuity(D_h, D_l, k=20)
    print(f"co_full = {co_full}, time = {perf_counter() - t0}")

    print("Same?", "Yes" if co_full == co_chunks else "No")
