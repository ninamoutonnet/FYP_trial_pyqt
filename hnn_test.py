import hnccorr.movie
from hnccorr import HNCcorr, Movie, HNCcorrConfig
from hnccorr.example import load_example_data
import tifffile
import numpy


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

print(stack.dtype)


movie = Movie('exp1', stack)

# config = HNCcorrConfig(postprocessor_min_cell_size = 40, postprocessor_preferred_cell_size = 80,postprocessor_max_cell_size = 200, patch_size = 21)
H = HNCcorr.from_config()  # Initialize HNCcorr with default configuration
H.segment(movie)    # perform the decomposition algorithm

H.segmentations  # List of identified cells
output = H.segmentations_to_list()  # Export list of cells (for Neurofinder)'''

print(output)
print(output.dtype)