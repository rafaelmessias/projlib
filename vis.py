import matplotlib.pyplot as plt
import numpy as np

def scatter(proj, labels=None, fig_size=None):
    if type(proj) == np.ndarray and proj.shape[1] == 2:
        fig, _ = plt.subplots()
        if fig_size:
            fig.set_size_inches(*fig_size)
        if type(labels) == list:
            if type(labels[0]) == str:                
                idx = list(set(labels))
                cmap = "tab10" if len(idx) <= 10 else "tab20"
                labels = [idx.index(x) for x in labels]
            plt.scatter(*proj.T, c=labels, cmap=plt.get_cmap(cmap, len(idx)))
            cbar = plt.colorbar(ticks=np.linspace(0.5, len(idx) - 0.5, len(idx) + 1))
            cbar.ax.set_yticklabels(idx)
        else:
            plt.scatter(*proj.T)
        
