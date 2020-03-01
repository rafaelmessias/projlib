import numpy as np
from scipy.spatial.distance import cdist


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


class Continuity:
    def __partial(self, D_high, D_low, k):
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

    def __aggregate(self, partial_results, n, k):
        total = sum(partial_results)
        return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * total)).squeeze())

    def compute(self, X, P, k=7, chunk_size=None):
        n = X.shape[0]
        if not chunk_size:
            chunk_size = n
        partial_results = []
        for i in range(0, n, chunk_size):
            D_h = cdist(X[i:i+chunk_size], X)
            D_l = cdist(P[i:i+chunk_size], P)
            partial_results.append(self.__partial(D_h, D_l, k))
        return self.__aggregate(partial_results, n, k)

if __name__ == "__main__":    
    from time import perf_counter
    import subprocess
    
    n = 1000
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)
    k = int(np.sqrt(n))

    np.savetxt("X.dat", X, fmt="%.10f", header=str(X.shape[1]), comments="")
    np.savetxt("P.dat", P, fmt="%.10f", header=str(P.shape[1]), comments="")
    p = subprocess.run([
            "./measure",
            "--datafile", "../../quality/X.dat",
            "--projfile", "../../quality/P.dat",
            "--neighbors", str(k)
        ], 
        cwd="../ext/dredviz-1.0.2", capture_output=True)
    print(float(p.stdout.split(b"\n")[-2].split()[5]))

    t0 = perf_counter()
    co_chunks = Continuity().compute(X, P, k=k, chunk_size=10)
    print(f"co_chunks = {co_chunks}, time = {perf_counter() - t0}")
    print(round(1.0 - co_chunks, 6))

    from scipy.spatial.distance import squareform, pdist
    t0 = perf_counter()
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))    
    co_full = continuity(D_h, D_l, k=k)
    print(f"co_full = {co_full}, time = {perf_counter() - t0}")
    print(round(1.0 - co_full, 6))

    print("Same?", "Yes" if co_full == co_chunks else "No")
