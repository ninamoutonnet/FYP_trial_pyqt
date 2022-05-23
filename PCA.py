import hyperspy.api as hs
import tifffile
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dask.array as da
from hyperspy.exceptions import LazyCupyConversion


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

        # Perform a signal decomposition using SVD
        with tifffile.Timer():
            s.change_dtype('float32')

        with tifffile.Timer():
            s.decomposition(algorithm="SVD", centre="navigation")

        _ = s.plot_explained_variance_ratio()

        # Plot the first four principle components (loadings and factors)
        # image_decomposition_loadings = s.plot_decomposition_loadings(9)
        # image_decomposition_loadings.savefig("decomposition_loadings.png") # save as png
        # image_decomposition_factors = s.plot_decomposition_factors(9)

        # get the results of the decomposition
        factors = s.learning_results.factors
        loadings = s.learning_results.loadings
        print(loadings.shape)

        comp_ids = range(9)

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)

        shape = s.axes_manager._signal_shape_in_array
        background = to_numpy(factors[:, 0].reshape(shape))
        first_pca = to_numpy(factors[:, 1].reshape(shape))
        difference = first_pca - background

        im = ax.imshow(background, cmap=matplotlib.cm.gray, interpolation='nearest', extent=None)
        plt.colorbar(im)
        plt.show()

        return
