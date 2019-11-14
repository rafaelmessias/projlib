import param

# TODO why does it not work with just import bhtsne...?
# TODO must check if the modules are installed before/while importing
from bhtsne import bhtsne
from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE as skTSNE
from OptSNE import OptSNE
from fitsne import FItSNE

class TSNE(param.Parameterized):
    perplexity = param.Number(50, bounds=(5, 75))
    early_exaggeration = param.Number(25, bounds=(5, 50))
    learning_rate = param.Number(200, bounds=(50, 400))
    # TODO verify if each import was successful before creating the list
    variant = param.ObjectSelector("bhtsne", 
        objects=[
            "bhtsne", "multicore", "sklearn", "optsne", "fitsne"
        ])

    def fit_transform(self, X):
        if self.variant == "bhtsne":            
            return bhtsne.run_bh_tsne(X, perplexity=self.perplexity, initial_dims=X.shape[1])
        
        if self.variant == "multicore":
            return MulticoreTSNE(n_jobs=4, perplexity=self.perplexity).fit_transform(X)

        if self.variant == "sklearn":
            return skTSNE(perplexity=self.perplexity).fit_transform(X)

        if self.variant == "optsne":
            return OptSNE(perplexity=self.perplexity).fit_transform(X)

        if self.variant == "fitsne":
            return FItSNE(X.values, perplexity=self.perplexity)

        return None