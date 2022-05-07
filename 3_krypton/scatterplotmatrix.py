"""
This module provides a couple utilities for creating the subplots used in
scatterplot matrixes (and simliar plots).
Created in June 2019, author: Sean T. Smith
"""

from numpy import linspace, meshgrid, sqrt
from numpy.random import rand, normal
import matplotlib.pyplot as plt


def scatterplot_matrix(x, labels, ax_label_font=14, plot_type='scatter',
                       fig_options={}, marginal_options={}, joint_options={}):
    ndim, nsamples = x.shape
    if type(fig_options) is tuple:
        fig, axes = fig_options
    else:
        fig, axes = plt.subplots(ndim, ndim, sharex='col', sharey='row',
                    gridspec_kw=dict(wspace=0, hspace=0), **fig_options)
        # Row & column formatting
        for i in range(ndim):
            axes[i][0].set_ylabel(labels[i], fontsize=ax_label_font)
            axes[i][0].set_ylim([x[i].min(), x[i].max()])
        for j in range(ndim):
            axes[-1][j].set_xlabel(labels[j], fontsize=ax_label_font)
            axes[-1][j].set_xlim([x[j].min(), x[j].max()])
        # Remove unwanted frames & ticks from the upper triangle
        for i in range(ndim-1):
            for j in range(i+1, ndim):
                axes[i][j].spines['top'].set_visible(False)
                axes[i][j].spines['bottom'].set_visible(False)
                axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].tick_params(axis='both', which='both',
                                       left=False, bottom=False)
    # Marginals
    nbins = max(min(nsamples // 75, 75), 10)
    for i in range(ndim):
        ax = axes[i][i].twinx()
        ax.hist(x[i], bins=nbins, density=True, **marginal_options)
        ax.set_ylim([0, None])
    axes[0][0].tick_params(axis='y', which='both',
                           left=False, right=False, labelleft=False)
    # Pairwise plots:
    nbins = max(min(int(sqrt(nsamples / 25)), 50), 10)
    for i in range(ndim):
        for j in range(i):
            ax = axes[i][j]
            if plot_type == 'scatter':
                ax.scatter(x[j], x[i], **joint_options)
            elif plot_type == 'hist':
                ax.hist2d(x[j], x[i], bins=nbins, **joint_options)
            elif plot_type == 'contour':
                xbins = linspace(x[j].min(), x[j].max(), nbins + 1)
                ybins = linspace(x[i].min(), x[i].max(), nbins + 1)
                freq, _, _, im = ax.hist2d(x[j], x[i], bins=[xbins, ybins])
                X, Y = meshgrid(xbins[:-1], ybins[:-1], indexing='xy')
                ax.contour(X, Y, freq.T, **joint_options)
                im.set_visible(False)
    return fig, axes


def contour_matrix(pdf, x_grids, labels, ax_label_font=14,
                   fig_options={}, marginal_options={}, joint_options={}):
    ndim = len(labels)
    if type(fig_options) is tuple:
        fig, axes = fig_options
    else:
        fig, axes = plt.subplots(ndim, ndim, sharex='col', sharey='row',
                    gridspec_kw=dict(wspace=0, hspace=0), **fig_options)
        # Row & column formatting
        for i in range(ndim):
            axes[i][0].set_ylabel(labels[i], fontsize=ax_label_font)
            axes[i][0].set_ylim([x_grids[i][0], x_grids[i][-1]])
        for j in range(ndim):
            axes[-1][j].set_xlabel(labels[j], fontsize=ax_label_font)
            axes[-1][j].set_xlim([x_grids[j][0], x_grids[j][-1]])
        # Remove unwanted frames & ticks from the upper triangle
        for i in range(ndim-1):
            for j in range(i+1, ndim):
                axes[i][j].spines['top'].set_visible(False)
                axes[i][j].spines['bottom'].set_visible(False)
                axes[i][j].spines['left'].set_visible(False)
                axes[i][j].spines['right'].set_visible(False)
                axes[i][j].tick_params(axis='both', which='both',
                                       left=False, bottom=False)
    # Marginals
    for i in range(ndim):
        marginal = pdf.copy()
        for k in range(i):
            shape = (-1,) + (1,) * (ndim - (k + 1))
            Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
            marginal = 0.5 * (Δxk * marginal[:-1] +
                              Δxk * marginal[+1:]).sum(axis=0)
        for k in range(i + 1, ndim):
            shape = (1, -1) + (1,) * (ndim - (k + 1))
            Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
            marginal = 0.5 * (Δxk * marginal[:, :-1] +
                              Δxk * marginal[:, +1:]).sum(axis=1)
        ax = axes[i][i].twinx()
        ax.plot(x_grids[i], marginal, **marginal_options)
        ax.set_ylim([0, None])
    axes[0][0].tick_params(axis='y', which='both',
                           left=False, right=False, labelleft=False)
    # Pairwise plots:
    for i in range(ndim):
        for j in range(i):
            joint = pdf.copy()
            for k in range(j):
                shape = (-1,) + (1,) * (ndim - (k + 1))
                Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
                joint = 0.5 * (Δxk * joint[:-1] +
                               Δxk * joint[+1:]).sum(axis=0)
            for k in range(j + 1, i):
                shape = (1, -1) + (1,) * (ndim - (k + 1))
                Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
                joint = 0.5 * (Δxk * joint[:, :-1] +
                               Δxk * joint[:, +1:]).sum(axis=1)
            for k in range(i + 1, ndim):
                shape = (1, 1, -1) + (1,) * (ndim - (k + 1))
                Δxk = (x_grids[k][1:] - x_grids[k][:-1]).reshape(shape)
                joint = 0.5 * (Δxk * joint[:, :, :-1] +
                               Δxk * joint[:, :, +1:]).sum(axis=2)
            X1, X2 = meshgrid(x_grids[j], x_grids[i], indexing='xy')
            ax = axes[i][j]
            ax.contour(X1, X2, joint.T, **joint_options)
    return fig, axes


if __name__ == '__main__':

    labels = ['a', 'b', 'c', 'd', 'e']
    nsamples = 100000
    x = normal(0, 1, (len(labels), nsamples))
    fig, axes = scatterplot_matrix(x, labels, plot_type='contour',
                                   fig_options=dict(figsize=(8, 8)))

    plt.show()
