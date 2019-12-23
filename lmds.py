import numpy as np
from scipy.spatial.distance import cdist
import param
from LMDS.mds import landmark_MDS

# TODO:
#   * Allow precomputed distances;
#   * Allow precomputed landmarks;
#   * Allow other distance metrics;
#   * Sanity checks for all params;

class LMDS(param.Parameterized):
    n_components = param.Number(2)
    landmarks = param.Number(0)
    
    def fit_transform(self, X):
        if self.landmarks <= 0:
            self.landmarks = int(np.sqrt(X.shape[0]))
        self.landmarks = np.random.choice(range(X.shape[0]), 
            size=self.landmarks, replace=False)
        D = cdist(X[self.landmarks], X)
        return landmark_MDS(D, self.landmarks, self.n_components)

if __name__ == "__main__":
    X = np.random.rand(1000, 10)
    P = LMDS().fit_transform(X)
    print(P.shape)
