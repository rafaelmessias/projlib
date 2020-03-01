import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from metric import ChunkedMetric

def neighborhood_hit(X, y, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))

# TODO it's not X, it's P (for projection)

class NeighborhoodHit(ChunkedMetric):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = X.shape[0]

    def partial(self, X, **kwargs):
        # I need the entire y for this operation
        neighbors = self.y[self.knn.kneighbors(X, return_distance=False)]
        # The first column of `neighbors` contains the labels of the points themselves
        labels = neighbors[:, 0].reshape(-1, 1)
        # Broadcast instead of tiling (as we did before)
        return np.sum(np.equal(neighbors, labels).astype('uint8'))

    # NOTE the returning value of get_chunk currently must be a tuple
    def get_chunk(self, start, chunk_size):
        return (self.X[start:start+chunk_size], )

    def aggregate(self, partial_results, k):
        n = self.X.shape[0]
        return np.sum(partial_results) / (n * k)

    def compute(self, chunk_size=None, chunk_search=True, k=7):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X, self.y)

        return super().compute(chunk_size=chunk_size, chunk_search=chunk_search, k=k)


if __name__ == "__main__":
    import random
    from time import perf_counter
    
    n = random.randint(500, 1000)
    P, y = np.random.rand(n, 2), np.random.randint(3, size=n)

    t0 = perf_counter()
    nh_chunks = NeighborhoodHit(P, y).compute(k=7, chunk_size=50)
    print(f"nh_chunks = {nh_chunks}, time = {perf_counter() - t0}")    
    
    t0 = perf_counter()
    nh_full = neighborhood_hit(P, y, k=7)
    print(f"nh_full = {nh_full}, time = {perf_counter() - t0}")

    # Equal up to 8 decimals
    print("Same?", "Yes" if np.isclose(nh_chunks, nh_full) else "No")
