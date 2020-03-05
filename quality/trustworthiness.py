import numpy as np
from scipy.spatial.distance import cdist
from projlib.quality.metric import DistanceBasedMetric


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
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


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
    
    n = random.randint(1000, 5000)
    X, P = np.random.rand(n, 10), np.random.rand(n, 2)

    t0 = perf_counter()
    tw_chunks = Trustworthiness().compute(X, P, k=20)
    print(f"tw_chunks = {tw_chunks}, time = {perf_counter() - t0}")
    
    t0 = perf_counter()    
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))
    tw_full = trustworthiness(D_h, D_l, k=20)
    print(f"tw_full = {tw_full}, time = {perf_counter() - t0}")

    # Equal up to 8 decimals
    print("Same?", "Yes" if np.isclose(tw_chunks, tw_full) else "No")
