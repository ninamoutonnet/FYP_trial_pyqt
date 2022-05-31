import os
import shutil

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

        # save the footprints and temporal traces of each neurons as a png in a dedicated folder
        # make the dedicated folder and if already exists, delete the previous one
        dir_general = 'RESULTS_' + str(filename)
        dir_general, temp = os.path.splitext(dir_general)  # removes the .tif
        if os.path.exists(dir_general):
            pass
        else:
            # the general directory does not exist, so create it
            os.makedirs(dir_general)

        dir = str(dir_general) + '/PCA_results'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        # Load the smaller tiff as s using the hyperspy library
        s = hs.load(filename)
        print(s)

        # Perform PCA a signal decomposition using SVD and deducting the mean along the navigation axis
        with tifffile.Timer():
            s.change_dtype('float32')

        with tifffile.Timer():
            s.decomposition(algorithm="SVD", centre="navigation")

        # get the results of the decomposition
        factors = s.learning_results.factors
        loadings = s.learning_results.loadings
        loadings = to_numpy(loadings)

        ####################
        ### REPORT PLOTS ###
        ####################
        '''image_decomposition_factors = s.plot_decomposition_factors(9)
        image_decomposition_factors.savefig("image_decomposition_factors_9.png", format="png", dpi=1200)
        
        fig, axs = plt.subplots(3, 3, figsize=(10,10))
        fig.suptitle("Loadings of the first 9 principal components", fontsize=16)
        axs[0, 0].plot(loadings[:, 0], linewidth = 0.7)
        axs[0, 0].set_title('Principal component, 0')
        axs[0, 1].plot(loadings[:, 1], linewidth = 0.7)
        axs[0, 1].set_title('Principal component, 1')
        axs[0, 2].plot(loadings[:, 2], linewidth = 0.7)
        axs[0, 2].set_title('Principal component, 2')
        axs[1, 0].plot(loadings[:, 3], linewidth = 0.7)
        axs[1, 0].set_title('Principal component, 3')
        axs[1, 1].plot(loadings[:, 4], linewidth = 0.7)
        axs[1, 1].set_title('Principal component, 4')
        axs[1, 2].plot(loadings[:, 5], linewidth = 0.7)
        axs[1, 2].set_title('Principal component, 5')
        axs[2, 0].plot(loadings[:, 6], linewidth = 0.7)
        axs[2, 0].set_title('Principal component, 6')
        axs[2, 1].plot(loadings[:, 7], linewidth = 0.7)
        axs[2, 1].set_title('Principal component, 7')
        axs[2, 2].plot(loadings[:, 8], linewidth = 0.7)
        axs[2, 2].set_title('Principal component, 8')
        for ax in axs.flat:
            ax.set(xlabel='frame number', ylabel='A.U')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig("PCA_loadings_9.png", format="png", dpi=1200)
        plt.show()'''

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
            loadings2 = np.copy(loadings)
            mean = np.mean(loadings2[:, i])
            loadings2[:, i].sort()
            strictly_above = len(loadings2[:, i]) - bisect(loadings2[:, i], mean)
            below_or_equal = len(loadings2[:, i]) - strictly_above
            diff[i] = strictly_above - below_or_equal

        ####################
        ### REPORT PLOTS ###
        ####################
        '''plt.rcParams["figure.figsize"] = (10, 7.5)
        plt.plot(diff)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('principal component index')
        plt.ylabel('(number of values above average - number of values below average)')
        plt.title('Plot showing, for every principal component''s loading, the difference between the '
                  'number \n of values above the loading''s average and the number of values below the loading''s average')
        plt.grid()
        plt.xlim((0, 10))
        plt.savefig("PCA_2.png", format="png", dpi=1200)
        plt.show()'''

        boundarie = np.where(diff >= 0)
        if len(boundarie) > 0 and len(boundarie[0]) > 0:
            # assume you always enter the loop
            limit = boundarie[0][0]
            print('First Index of element >=0 ', limit)

        for neuron_index in range(limit-1):

            neuron_index = neuron_index + 1 # avoid i=0
            fig, axes = plt.subplots(1, 2, figsize=(10, 2), gridspec_kw={'width_ratios': [1, 3]})

            # spatial footprint
            shape = s.axes_manager._signal_shape_in_array
            factor_im = to_numpy(factors[:, neuron_index].reshape(shape))
            axes[0].imshow(factor_im)
            axes[0].set_title("PCA factor, spatial footprint", fontsize=5)
            axes[0].grid(False)
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            # temporal dynamics
            axes[1].set_title("Loading factor", fontsize=5)
            axes[1].plot(loadings[:, neuron_index], label='neuron {0}'.format(neuron_index), color='blue')
            axes[1].set_ylabel("A.U", fontsize=5)
            axes[1].set_xlabel("frame number", fontsize=5)
            axes[1].grid()

            name = dir + '/PCA_results_' + str(neuron_index)
            plt.savefig(name, dpi=1200)
            # allows to not display the individual ones
            plt.close(fig)

        print('done')
        # shape = s.axes_manager._signal_shape_in_array
        # first_pca = to_numpy(factors[:, 1].reshape(shape))
        # im = ax.imshow(first_pca, cmap=matplotlib.cm.gray, interpolation='nearest', extent=None)
        # plt.colorbar(im)
        # plt.show()

        return
