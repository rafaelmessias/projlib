import param

VARIANTS = []

try:
    import bhtsne
    VARIANTS.append("bhtsne")
except:
    pass

try:
    from MulticoreTSNE import MulticoreTSNE
    VARIANTS.append("multicore")
except:
    pass

try:
    from sklearn.manifold import TSNE as skTSNE
    VARIANTS.append("sklearn")
except:
    pass

try:
    from OptSNE import OptSNE
    VARIANTS.append("optsne")
except:
    pass

try:
    from fitsne import FItSNE
    VARIANTS.append("fitsne")
except:
    pass

try:
    from tsnecuda import TSNE as CudaTSNE
    VARIANTS.append("cuda")
except:
    pass

class TSNE(param.Parameterized):
    perplexity = param.Number(50, bounds=(5, 75))
    early_exaggeration = param.Number(25, bounds=(5, 50))
    learning_rate = param.Number(200, bounds=(50, 400))
    variant = param.ObjectSelector(VARIANTS[0], objects=VARIANTS)

    def fit_transform(self, X):
        if self.variant == "bhtsne":            
            return bhtsne.tsne(X, perplexity=self.perplexity)
        
        if self.variant == "multicore":
            return MulticoreTSNE(n_jobs=4, perplexity=self.perplexity).fit_transform(X)

        if self.variant == "sklearn":
            return skTSNE(perplexity=self.perplexity).fit_transform(X)

        if self.variant == "optsne":
            return OptSNE(perplexity=self.perplexity).fit_transform(X)

        if self.variant == "fitsne":
            return FItSNE(X, perplexity=self.perplexity)

        if self.variant == "cuda":
            return CudaTSNE(perplexity=self.perplexity).fit_transform(X)

        return None