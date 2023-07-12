import matplotlib.pyplot as plt
import numpy as np


def pre_post_with_confidence(data, x='mean_latitude', y='altitude', pre_name='linear_pre', post_name='linear_post',
                             figsize=(10, 6), clim=10, contour=True, ylim=None, log_y=False, x_label=None, y_label=None,
                             pre_title=None, post_title=None, c_label=None):
    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)

    plot_with_confidence(data, pre_name, x, y, clim, contour=contour)
    if ylim is not None:
        plt.ylim(ylim)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if pre_title is not None:
        plt.title(pre_title)

    if log_y:
        plt.yscale('log')
    plt.subplot(1, 2, 2)

    im = plot_with_confidence(data, post_name, x, y, clim, contour=contour)
    if ylim is not None:
        plt.ylim(ylim)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if post_title is not None:
        plt.title(post_title)

    if log_y:
        plt.yscale('log')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    if c_label:
        fig.colorbar(im, cax=cbar_ax, label=c_label)
    else:
        fig.colorbar(im, cax=cbar_ax)


def plot_with_confidence(data, name, x='mean_latitude', y='altitude', clim=10, contour=True):
    levels = np.concatenate(([-clim*50], np.linspace(-clim, clim), [clim*50]))

    x_diff = data[x].values[1] - data[x].values[0]

    xs = np.hstack((data[x].values, (data[x].values[-1] + x_diff))) - x_diff / 2

    if len(data[y].values) == np.shape(data[name].values)[1]:
        vals = data[name].values.T
    else:
        vals = data[name].values

    if contour:
        C = plt.pcolor(data[x].values, data[y].values, vals, cmap='RdBu_r', alpha=0)

        im = plt.contourf(data[x].values, data[y].values, vals, cmap='RdBu_r', levels=levels)
    else:
        C = plt.pcolor(data[x].values, data[y].values, vals, cmap='RdBu_r')
        im = C


    plt.clim(-clim, clim)
    # Convert to 95%
    error = data[name + '_std'] * 2

    significant = np.abs(data[name]) > error

    for idx, p in enumerate(C._paths):
        mean_lat = np.nanmean(np.unique(p._vertices[:, 0]))
        mean_alt = np.nanmean(np.unique(p._vertices[:, 1]))

        mask = significant.sel({x: mean_lat, y:mean_alt}, method='nearest').values

        if ~mask:
            # Hatching doesnt seem to work very well
            # ax.add_patch(Polygon(p._vertices, hatch='xxxx', color='blue', alpha=0.2, lw=0, fill=False))

            lats = np.unique(p._vertices[:, 0])
            alts = np.unique(p._vertices[:, 1])

            plt.plot(lats, alts, 'k', linewidth=0.1)
            plt.plot(lats, alts[::-1], 'k', linewidth=0.1)
    return im


def post_with_confidence(data, x = 'mean_latitude', y = 'altitude', post_name = 'linear_post',
    figsize = (10, 6), clim = 10, contour = True, ylim = None, log_y = False, x_label = None, y_label = None,
    post_title = None, c_label = None, fig=None):
    if fig is None:
        fig = plt.figure(figsize=figsize)

    im = plot_with_confidence(data, post_name, x, y, clim, contour=contour)
    if ylim is not None:
        plt.ylim(ylim)

    if x_label is not None:
        plt.xlabel(x_label)

    if y_label is not None:
        plt.ylabel(y_label)

    if post_title is not None:
        plt.title(post_title)

    if log_y:
        plt.yscale('log')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    if c_label:
        fig.colorbar(im, cax=cbar_ax, label=c_label)
    else:
        fig.colorbar(im, cax=cbar_ax)