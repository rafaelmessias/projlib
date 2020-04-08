from time import perf_counter    
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from joblib import Parallel, delayed


# Differently than the previous version, this one computes the correlation with the full
#   square matrix. I could not find a way to break the triangle matrix into chunks.
def shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high.flatten(), D_low.flatten())[0]


# TODO I can probably save a bit of time/space if I manage to process/store only the triangle distance matrices
# TODO maybe the final step of the rs calculation can also be parallelized
# TODO add the option to not parallelize this, not to use memmap, and to set the temp_dir
# TODO can I parallelize the sorting also?
# TODO all the loops that use chunks can be parallelized

class ShepardCorrelation:
    
    def compute(self, X, P, chunk_size=None):
        n = X.shape[0]

        if not chunk_size:
            chunk_size = n

        # These will store the intermediate distance matrices
        tmp_dir = "/mnt/external/tmp/"
        # tmp_dir = ""

        def func(i, X):
            t0 = perf_counter()

            # D = pairwise_distances(X, n_jobs=-1)

            dtype=[('values', 'f8'), ('indices', 'u8')]
            D = np.memmap(tmp_dir + f"D_{i}.memmap", dtype=dtype, mode="w+", shape=(X.shape[0]**2,))
            for j in range(0, X.shape[0]**2, chunk_size):
                D["indices"][j:j+chunk_size] = np.arange(j, j+chunk_size)
        
            row_ptr = 0
            for chunk in pairwise_distances_chunked(X, working_memory=chunk_size):
                length = chunk.ravel().shape[0]
                D["values"][row_ptr:row_ptr+length] = chunk.ravel()
                row_ptr += length

            t1 = perf_counter() - t0
            print(f"Done with distances ({i}) t={t1}")
            t0 = perf_counter()

            D.sort(order='values', axis=0)

            t1 = perf_counter() - t0
            print(f"Done with sorting ({i}) t={t1}")
            t0 = perf_counter()

            for j in range(0, X.shape[0]**2, chunk_size):
                idx = D["indices"][j:j+chunk_size]
                D["values"][idx] = np.arange(j+1, j+chunk_size+1)            
            
            ranked = D["values"]
                        
            # np.float64 avoids overflows in the computation that comes next
            # ranked = stats.rankdata(D["values"], method="ordinal").astype(np.float64)

            t1 = perf_counter() - t0
            print(f"Done with ranking ({i}) t={t1}")

            return ranked

        # a, b = func(X), func(P)
        p = Parallel(n_jobs=2)
        a, b = p((delayed(func))(i, x) for i, x in enumerate([X, P]))

        print(f"Final sum")

        # num = 6 * np.sum((a - b)**2)
        a -= b
        a **= 2
        num = 6 * a.sum()
        den = (n**6 - n**2)
        rs = 1 - num / den
        return rs

        # rs = np.corrcoef(a_ranked, b_ranked)        
        # return rs[1, 0]

        # return spearmanr(D_h.flatten(), D_l.flatten())


# TODO test these things instead with artificial clustered data, with random noise
# TODO refactor out a standard benchmark for all the metrics

if __name__ == "__main__":
    import random    
    
    n = 3000
    X, P = np.random.rand(n, 100), np.random.rand(n, 2)

    t0 = perf_counter()
    sc_chunks = ShepardCorrelation().compute(X, P, chunk_size=100)
    print(f"sc_chunks = {np.round(sc_chunks, 8)}, time = {perf_counter() - t0}")
        
    from scipy.spatial.distance import squareform, pdist
    t0 = perf_counter()
    D_h, D_l = pairwise_distances(X, n_jobs=-1), pairwise_distances(P, n_jobs=-1)
    print("SH: Done with distances; now for the Spearman's R calculation")
    sc_full = shepard_diagram_correlation(D_h, D_l)
    print(f"sc_full = {np.round(sc_full, 8)}, time = {perf_counter() - t0}")
    
    # Equal up to 8 decimals
    print("Same?", "Yes" if np.isclose(sc_chunks, sc_full) else "No")
