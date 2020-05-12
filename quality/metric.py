import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed


# TODO right now the abstract class assumes that I want to compute chunked distances; should be more general.
# TODO at some point I must stop trying new chunk sizes and just stick with what is working.
# TODO multiple jobs only work without chunk search, for now 
# TODO 'self' is being passed to every process, which could potentially be slow (due to transfer of
#      data between parent and child). Try to improve on this?

def _run_chunk(self, i, chunk_size, **kwargs):
    # print(f"_run_chunk: {i}, size: {chunk_size}")
    chunk = self.get_chunk(i, chunk_size)
    return self.partial(*chunk, **kwargs)


class ChunkedMetric:

    def partial(self, *chunk, **kwargs):
        raise NotImplementedError("Must override 'partial'")

    def aggregate(self, partial_results, **kwargs):
        raise NotImplementedError("Must override 'aggregate'")

    def get_chunk(self, start, chunk_size):
        raise NotImplementedError("Must override 'get_chunk'")

    def dataset_size(self):
        raise NotImplementedError("Must override 'dataset_size'")
    
    def compute(self, chunk_size=100, chunk_search=False, n_jobs=1, **kwargs):
        n = self.dataset_size()
        partial_results = []

        if chunk_search:
            if not chunk_size:
                chunk_size = (n + 1) // 2
            
            i = 0
            low, hi = 1, n
            while i < n:
                print(f"Chunk start: {i} (of {n}), size: {chunk_size} ({low}, {hi})")
                try:
                    partial_results.append(_run_chunk(self, i, chunk_size, **kwargs))
                    # Break if you want to debug memory usage of each chunk
                    # break
                    # Success; move on with the next chunk
                    i += chunk_size
                    # Maybe we can raise the chunk_size? 
                    # But don't search too much (stop if hi and low are close to each other).
                    # if (hi - low) > 1000:
                    #     low = chunk_size

                except MemoryError:
                    # Failure; the chunk_size is too large
                    hi = chunk_size - 1                    
                    # i is not incremented
                    print("Out of memory; trying with smaller chunk.")
                
                chunk_size = (low + hi) // 2
                
        else:
            if not chunk_size:
                chunk_size = n

            p = Parallel(n_jobs=n_jobs)
            partial_results = p((delayed(_run_chunk))(self, i, chunk_size, **kwargs) for i in range(0, n, chunk_size))
        
        return self.aggregate(partial_results, **kwargs)


class DistanceBasedMetric(ChunkedMetric):

    # TODO implement a cache here for the cases when the current chunk is shrinked
    def get_chunk(self, start, chunk_size):
        D_h = cdist(self.X[start:start+chunk_size], self.X, metric="sqeuclidean")
        D_l = cdist(self.P[start:start+chunk_size], self.P, metric="sqeuclidean")
        return D_h, D_l

    def dataset_size(self):
        return self.X.shape[0]

    # TODO add an option to pass the distance matrix as input instead
    def compute(self, X, P, **kwargs):
        self.X = X
        self.P = P        
        return super().compute(**kwargs)
