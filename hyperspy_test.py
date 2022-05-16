# Set the matplotlib backend
import hyperspy_gui_traitsui
import hyperspy_gui_ipywidgets
import matplotlib.pyplot
import tifffile
import hyperspy.api as hs
import cv2  # OpenCV for fast interpolation
from PIL import Image
import numpy

hs.preferences.gui()

downsampling = 8  # dividing factor, 4 yields a 512x512 if the input is 2048x2048

filename = 'im1.tiff'

with tifffile.Timer():
    stack = tifffile.imread(filename)[:, ::downsampling, ::downsampling].copy()

with tifffile.Timer():
    with tifffile.TiffFile(filename) as tif:
        page = tif.pages[0]
        shape = len(tif.pages), page.imagelength // downsampling, page.imagewidth // downsampling
        stack = numpy.empty(shape, page.dtype)
        for i, page in enumerate(tif.pages):
            stack[i] = page.asarray(out='memmap')[::downsampling, ::downsampling]
            # # better use interpolation instead:
            # stack[i] = cv2.resize(
            # page.asarray(),
            # dsize=(shape[2], shape[1]),
            # interpolation=cv2.INTER_LINEAR)

print(stack.shape)

# save the downsampled numpy array to a tiff file
imlist = []
with tifffile.Timer():
    for m in stack:
        imlist.append(Image.fromarray(m))

    imlist[0].save("test.tif", compression="tiff_deflate", save_all=True,
                   append_images=imlist[1:])

# Load the smaller tiff as s using the hyperspy library
s = hs.load("test.tif")

# Perform a signal decomposition using SVD
with tifffile.Timer():
    s.change_dtype('float32')
    #  s.decomposition()  # 3.574203 s
    #  s.decomposition(True, algorithm="NMF", output_dimension=9)  # 13.727829 s
    #  s.decomposition(normalize_poissonian_noise=True) # 2.795001s
    # s.decomposition(False, algorithm="MLPCA", output_dimension=9)  #

with tifffile.Timer():
    # s.decomposition()
    # s.decomposition(algorithm="ORPCA", output_dimension=9, method="MomentumSGD",
# subspace_learning_rate=1.1, subspace_momentum=0.5) DOES NOT RUN
    #s.decomposition(True, algorithm="NMF", output_dimension=9)
    #    s.decomposition(algorithm="RPCA", output_dimension=3, lambda1=0.1)

    # s.decomposition(False, algorithm="MLPCA", output_dimension=9)  #

s.data.shape

print('DONE')

# Plot the "explained variance ratio" (scree plot) --> ONLY IN SVD AND PCA, not NMF
# _ = s.plot_explained_variance_ratio()

PCA_number = 9

# Plot the first four principle components (loadings and factors)
image_decomposition_loadings = s.plot_decomposition_loadings(PCA_number)
# image_decomposition_loadings.savefig("decomposition_loadings.png") # save as png

print('type is: ', type(image_decomposition_loadings))
print('type is: ', type(image_decomposition_loadings.patch))

image_decomposition_factors = s.plot_decomposition_factors(PCA_number)

# image_decomposition_factors.savefig("decomposition_factors.png") # save as png


matplotlib.pyplot.show()
