import numpy as np
from scipy.spatial.distance import cdist


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


class Trustworthiness:    

    def __partial(self, D_h, D_l, k):
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

    def __aggregate(self, partial_results, n, k):
        total = sum(partial_results)
        return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * total)).squeeze())
    
    def compute(self, X, P, k=7, chunk_size='search'):
        n = X.shape[0]
        partial_results = []

        if chunk_size == 'search':            
            i = 0
            low, hi = 1, n
            chunk_size = (low + hi) // 2
            while i < n:
                print(f"Chunk start: {i}, end: {i + chunk_size}")
                try:
                    D_h = cdist(X[i:i+chunk_size], X)
                    D_l = cdist(P[i:i+chunk_size], P)
                    partial_results.append(self.__partial(D_h, D_l, k))
                    # Break if you want to debug memory usage of each chunk
                    # break
                    # Success; move on with the next chunk
                    i += chunk_size
                    # Maybe we can raise the chunk_size?
                    low += chunk_size
                    chunk_size = (low + hi) // 2

                except MemoryError:                    
                    # Failure; the chunk_size is too large
                    hi = low + chunk_size
                    chunk_size = (low + hi) // 2
                    # i is not incremented
                    print("Memory error. New chunk size:", chunk_size) 
        else:
            if not chunk_size:
                chunk_size = n
            
            for i in range(0, n, chunk_size):
                print(f"Chunk start: {i}, end: {i + chunk_size}")
                D_h = cdist(X[i:i+chunk_size], X)
                D_l = cdist(P[i:i+chunk_size], P)
                partial_results.append(self.__partial(D_h, D_l, k))
                # Break if you want to debug memory usage of each chunk
                # break
        
        return self.__aggregate(partial_results, n, k)


if __name__ == "__main__":
    import subprocess
    from scipy.spatial.distance import squareform, pdist

    n = 1000
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)
    k=int(np.sqrt(n))

    np.savetxt("X.dat", X, fmt="%.10f", header=str(X.shape[1]), comments="")
    np.savetxt("P.dat", P, fmt="%.10f", header=str(P.shape[1]), comments="")
    p = subprocess.run([
            "./measure",
            "--datafile", "../../quality/X.dat",
            "--projfile", "../../quality/P.dat",
            "--neighbors", str(k)
        ], 
        cwd="../ext/dredviz-1.0.2", capture_output=True)
    print(float(p.stdout.split(b"\n")[-2].split()[2]))

    tw = Trustworthiness().compute(X, P, k=k)
    print(round(1.0 - tw, 6))
    
    D_h, D_l = squareform(pdist(X)), squareform(pdist(P))
    tw_full = trustworthiness(D_h, D_l, k=k)
    print(round(1.0 - tw_full, 6))

    # TODO remove X.dat and P.dat
