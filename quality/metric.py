import numpy as np
from scipy.spatial.distance import cdist


# TODO right now the abstract class assumes that I want to compute chunked distances; should be more general.
# TODO at some point I must stop trying new chunk sizes and just stick with what is working.

class ChunkedMetric:

    def partial(self, *chunk, **kwargs):
        raise NotImplementedError("Must override 'partial'")

    def aggregate(self, partial_results, **kwargs):
        raise NotImplementedError("Must override 'aggregate'")

    def get_chunk(self, start, chunk_size):
        raise NotImplementedError("Must override 'get_chunk'")
    
    def compute(self, chunk_size=None, chunk_search=True, **kwargs):
        partial_results = []

        if chunk_search:
            if not chunk_size:
                chunk_size = (self.n + 1) // 2
            
            i = 0
            low, hi = 1, self.n
            while i < self.n:
                print(f"Chunk start: {i} (of {self.n}), size: {chunk_size} ({low}, {hi})")
                try:
                    chunk = self.get_chunk(i, chunk_size)
                    partial_results.append(self.partial(*chunk, **kwargs))
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
                chunk_size = self.n
            
            for i in range(0, self.n, chunk_size):
                chunk = self.get_chunk(i, chunk_size)
                partial_results.append(self.partial(*chunk, **kwargs))
                # Break if you want to debug memory usage of each chunk
                # break
        
        return self.aggregate(partial_results, **kwargs)


class DistanceBasedMetric(ChunkedMetric):

    def __init__(self, X, P):
        self.X = X
        self.P = P
        self.n = X.shape[0]

    # TODO implement a cache here for the cases when the current chunk is shrinked
    def get_chunk(self, start, chunk_size):
        D_h = cdist(self.X[start:start+chunk_size], self.X)
        D_l = cdist(self.P[start:start+chunk_size], self.P)
        return D_h, D_l

