from matplotlib import pyplot as plt
import numpy as np

__copyright__ = "Copyright 2016-2017, Netflix, Inc."
__license__ = "Apache, Version 2.0"


def get_cdf(x, num_bins=100):
    x = np.array(x)
    counts, bin_edges = np.histogram(x, bins=num_bins)
    cdf = np.cumsum(counts)
    cdf = cdf / float(cdf[-1]) # normalize
    bin_edges = bin_edges[1:] # make size
    return cdf, bin_edges


def get_pdf(data, num_bins=20):
    pdf, bin_edges = np.histogram(data, density=True, bins=num_bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    return pdf, bin_centres


def plot_distribution(plot_type, df, key, slice_name, slices, colors=None):
    if colors is None:
        colors = [None for _ in slices]
    handles = []
    for slice, color in zip(slices, colors):
        if isinstance(slice, (list, tuple)):
            data = df.loc[df[slice_name].isin(slice)][key].tolist()
        else:
            data = df.loc[df[slice_name] == slice][key].tolist()
        if plot_type == 'cdf':
            ys, xs = get_cdf(data)
            plt.ylabel('CDF')
        elif plot_type == 'pdf':
            ys, xs = get_pdf(data)
            plt.ylabel('PDF')
        else:
            assert False, "Unknown plot type: {}".format(plot_type)
        if color:
            handle = plt.plot(xs, ys, label="{}".format(str(slice)), color=color)
        else:
            handle = plt.plot(xs, ys, label="{}".format(str(slice)))
        plt.grid(which='major')
        handles.append(handle)
    return handles
