import hyperspy.api as hs
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dask.array as da
from hyperspy.exceptions import LazyCupyConversion
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.stats import chisquare
from bisect import bisect

def is_cupy_array(array):
    """
    Convenience function to determine if an array is a cupy array
    Parameters
    ----------
    array : array
        The array to determine whether it is a cupy array or not.
    Returns
    -------
    bool
        True if it is cupy array, False otherwise.
    """
    try:
        import cupy as cp
        return isinstance(array, cp.ndarray)
    except ImportError:
        return False


def to_numpy(array):
    """
    Returns the array as a numpy array. Raises an error when a dask array is
    provided.
    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        Array to convert to numpy array.
    Returns
    -------
    array : numpy.ndarray

    Raises
    ------
    ValueError
        If the provided array is a dask array
    """
    if isinstance(array, da.Array):
        raise LazyCupyConversion
    if is_cupy_array(array):
        import cupy as cp
        array = cp.asnumpy(array)

    return array


class PCA_class():

    def __init__(self, filename):
        # Load the smaller tiff as s using the hyperspy library
        s = hs.load(filename)
        print(s)

        # Perform PCA a signal decomposition using SVD and deducting the mean along the navigation axis
        with tifffile.Timer():
            s.change_dtype('float32')

        with tifffile.Timer():
            s.decomposition(algorithm="SVD", centre="navigation")

        # _ = s.plot_explained_variance_ratio()

        # Plot the first four principle components (loadings and factors)
        # image_decomposition_loadings = s.plot_decomposition_loadings(9)
        # image_decomposition_loadings.savefig("decomposition_loadings.png") # save as png

        '''image_decomposition_factors = s.plot_decomposition_factors([ 1, 2, 3, 4, 5])
        image_decomposition_loadings = s.plot_decomposition_loadings([1])
        image_decomposition_loadings = s.plot_decomposition_loadings([2])
        image_decomposition_loadings = s.plot_decomposition_loadings([3])
        image_decomposition_loadings = s.plot_decomposition_loadings([4])
        image_decomposition_loadings = s.plot_decomposition_loadings([5])'''

        image_decomposition_factors = s.plot_decomposition_factors([1, 2, 3, 4, 5])
        image_decomposition_loadings = s.plot_decomposition_loadings([1, 2, 3, 4, 5])
        plt.show()

        # get the results of the decomposition
        factors = s.learning_results.factors
        loadings = s.learning_results.loadings
        loadings = to_numpy(loadings)

        '''# does not work - runs test
        z_value = np.zeros((loadings.shape[1], 1))
        a_value = np.zeros((loadings.shape[1], 1))
        for i in range(loadings.shape[1]):
            z_value[i] = runstest_1samp(loadings[:, i], correction=False)[0]
            a_value[i] = runstest_1samp(loadings[:, i], correction=False)[1]
        plt.plot(z_value)
        plt.show()
        plt.plot(a_value)
        plt.show()'''

        '''# does not work - chi square test
        chisq_value = np.zeros((loadings.shape[1], 1))
        p_value = np.zeros((loadings.shape[1], 1))
        for i in range(loadings.shape[1]):
            chisq_value[i] = chisquare(loadings[:, i])[0]
            p_value[i] = chisquare(loadings[:, i])[1]
        plt.plot(chisq_value)
        plt.show()
        plt.plot(p_value)
        plt.show()'''

        '''# does not work - std
        std = np.zeros((20, 1))
        print(std.shape)
        for i in range(20):
            std[i] = np.std(loadings[:, i])
        plt.plot(std)
        plt.show()'''

        # how many values are above/below the average?
        diff = np.zeros((loadings.shape[1], 1))
        for i in range(loadings.shape[1]):
            mean = np.mean(loadings[:, i])
            loadings[:, i].sort()
            strictly_above = len(loadings[:, i]) - bisect(loadings[:, i], mean)
            below_or_equal = len(loadings[:, i]) - strictly_above
            diff[i] = strictly_above - below_or_equal
        plt.plot(diff)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('principal component index')
        plt.ylabel('(number of values above average - number of values below average)')
        plt.title('Plot showing, for every principal component''s loading, the difference between the '
                  'number \n of values above the loading''s average and the number of values below the loading''s average')
        plt.grid()
        plt.show()
        plt.savefig("PCA_1.png", format="png", dpi=1200)


        '''f = plt.figure()
        ax = f.add_subplot(1, 1, 1)

        shape = s.axes_manager._signal_shape_in_array
        first_pca = to_numpy(factors[:, 1].reshape(shape))

        im = ax.imshow(first_pca, cmap=matplotlib.cm.gray, interpolation='nearest', extent=None)
        plt.colorbar(im)
        plt.show()'''

        return
