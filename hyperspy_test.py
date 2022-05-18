# Set the matplotlib backend
import hyperspy_gui_traitsui
import hyperspy_gui_ipywidgets
import matplotlib.pyplot
import numpy as np
from hyperspy.drawing import signal as sigdraw
import tifffile
import hyperspy.api as hs
import cv2  # OpenCV for fast interpolation
from PIL import Image
from hyperspy.exceptions import LazyCupyConversion
import numpy
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
import dask.array as da

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


def open_downsample_save_tiff(downsampling):
    #downsampling = 8  # dividing factor, 4 yields a 512x512 if the input is 2048x2048
    filename = 's2a4d1_WF_1P_1x1_400mA_50Hz_func_500frames_4AP_1_MMStack.tif'
    with tifffile.Timer():
        stack = tifffile.imread(filename)[:, ::downsampling, ::downsampling].copy()

    with tifffile.Timer():
        with tifffile.TiffFile(filename) as tif:
            page = tif.pages[0]
            shape = len(tif.pages), page.imagelength // downsampling, page.imagewidth // downsampling
            stack = numpy.empty(shape, page.dtype)
            for i, page in enumerate(tif.pages):
                # stack[i] = page.asarray(out='memmap')[::downsampling, ::downsampling]
                # better use interpolation instead:
                stack[i] = cv2.resize(page.asarray(), dsize=(shape[2], shape[1]), interpolation=cv2.INTER_LINEAR)
    print(stack.shape)

    # save the downsampled numpy array to a tiff file
    imlist = []
    with tifffile.Timer():
        for m in stack:
            imlist.append(Image.fromarray(m))

        imlist[0].save("test.tif", compression="tiff_deflate", save_all=True,
                       append_images=imlist[1:])

# open the tiff file
open_downsample_save_tiff(8)

# Load the smaller tiff as s using the hyperspy library
s = hs.load("test.tif")

# Perform a signal decomposition using SVD
with tifffile.Timer():
    s.change_dtype('float32')

with tifffile.Timer():
    # s.decomposition()
    s.decomposition(algorithm="SVD", centre="navigation")
    # s.decomposition(algorithm="ORPCA", output_dimension=9, method="MomentumSGD", subspace_learning_rate=1.1, subspace_momentum=0.5) DOES NOT RUN
    # s.decomposition(True, algorithm="NMF", output_dimension=9)
    # s.decomposition(algorithm="RPCA", output_dimension=3, lambda1=0.1)
    # s.decomposition(False, algorithm="MLPCA", output_dimension=9)  #


# Plot the "explained variance ratio" (scree plot) --> ONLY IN SVD AND PCA, not NMF
#_ = s.plot_explained_variance_ratio()

PCA_number = 6

# Plot the first four principle components (loadings and factors)
image_decomposition_loadings = s.plot_decomposition_loadings(PCA_number)
# image_decomposition_loadings.savefig("decomposition_loadings.png") # save as png
image_decomposition_factors = s.plot_decomposition_factors(comp_ids = PCA_number)
#print('type is: ', type(image_decomposition_factors))
#print('type is: ', type(image_decomposition_factors.patch))

################################################
######           REPLICATION          ##########
################################################

# get the results of the decomposition
factors = s.learning_results.factors
print('factors_shape: ', factors.shape )
# comp_ids = 9 = PCA_number
comp_ids = 9

# call self._plot_factors_or_pchars(factors,
#                                             comp_ids=comp_ids,
#                                             calibrate=calibrate,
#                                             same_window=same_window,
#                                             comp_label=title,
#                                             cmap=cmap,
#                                             per_row=per_row)


# default: def _plot_factors_or_pchars(self, factors, comp_ids=None,
#                                 calibrate=True, avg_char=False,
#                                 same_window=True, comp_label='PC',
#                                 img_data=None,
#                                 plot_shifts=True, plot_char=4,
#                                 cmap=plt.cm.gray, quiver_color='white',
#                                 vector_scale=1,
#                                 per_row=3, ax=None):


################################################
###### print the first 9 PCA components ########
######         USES SIGDRAW             ########
################################################

'''
comp_ids = 9
comp_ids = range(comp_ids)
f = matplotlib.pyplot.figure(figsize=(4 * 3, 3 * 3))

for i in range(len(comp_ids)):
    ax = f.add_subplot(3, 3, i + 1)
    sigdraw._plot_2D_component(factors=factors,
                               idx=comp_ids[i],
                               axes_manager=s.axes_manager,
                               calibrate=True, ax=ax,
                               cmap=matplotlib.cm.gray, comp_label='hi')'''


################################################
###### print the first   PCA component  ########
######     DOES NOT USE SIGDRAW         ########
################################################

'''
f = matplotlib.pyplot.figure()
ax = f.add_subplot(1, 1, 1)

shape = s.axes_manager._signal_shape_in_array
background = to_numpy(factors[:, 0].reshape(shape))
first_pca = to_numpy(factors[:, 1].reshape(shape))
second_pca = to_numpy(factors[:, 2].reshape(shape))
difference = first_pca-background

axes = s.axes_manager.signal_axes[::-1]
im = ax.imshow(difference, cmap=matplotlib.cm.gray, interpolation='nearest', extent=None)
matplotlib.pyplot.colorbar(im)'''


################################################
###### MASK FOR THE FIRST PCA components #######
################################################

number_to_keep = 6
shape = s.axes_manager._signal_shape_in_array
overall_pca_mask = np.ones(shape)

for k in range(number_to_keep+1): # the +1 removes the first component, which is the background
    first_pca = to_numpy(factors[:, k].reshape(shape))
    lower_bound = np.amin(first_pca)
    upper_bound = np.amax(first_pca)
    range_intensity = upper_bound - lower_bound
    ################################################
    #         MAKE THIS A SLIDER IN THE GUI        #
    ################################################
    threshold = 0.55*(range_intensity) + lower_bound
    print('thershold: ', threshold)
    shape2 = first_pca.shape
    first_pca_mask = np.zeros(shape2)
    for i in range(256):
        for j in range(256):
            if first_pca[i,j] < threshold:
                first_pca_mask[i,j] = 1
    overall_pca_mask = overall_pca_mask * first_pca_mask

overall_pca_mask = dilation(overall_pca_mask)
matplotlib.pyplot.imshow(overall_pca_mask, interpolation='nearest', cmap=matplotlib.pyplot.gray())
matplotlib.pyplot.show()
