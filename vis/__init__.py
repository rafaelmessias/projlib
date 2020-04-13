import matplotlib.pyplot as plt
import numpy as np

# TODO choose right colormap for everything up to a reasonable number of classes, e.g. 14
# TODO order the legend labels and fix the position of the ticks
# TODO better sizes for the figure and the markers
# TODO alpha?
# TODO The "spectral" colormap is probably nice for up to 10 categories
# TODO The colorbar looks too big depending on the fig_size, as it takes the entire height
#      Example: https://matplotlib.org/3.1.0/gallery/axes_grid1/demo_colorbar_with_inset_locator.html
# TODO The colorbar also does not work correctly with less colors than the length of cmap


def scatter(proj, labels=None, fig_size=None, title=None):
    if type(proj) == np.ndarray and proj.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'datalim')
        #plt.axis('off')
        if fig_size:
            fig.set_size_inches(*fig_size)
        else:
            height = 9
            # 1.125 is just a rough estimate for compensating the colorbar
            fig.set_size_inches((height * 1.125, height))
        if type(labels) == list or type(labels) == np.ndarray:
            idx = sorted(list(set(labels)))
            cmap = "tab10" if len(idx) <= 10 else "tab20"
            if type(labels[0]) == str:
                labels = [idx.index(x) for x in labels]
            plt.scatter(*proj.T, s=30, c=labels, cmap=cmap, edgecolors='k')
            plt.colorbar(ticks=idx)
            plt.clim([idx[0]-0.5, idx[-1]+0.5])
        else:
            plt.scatter(*proj.T)
        if type(title) == str:
            plt.title(title)
