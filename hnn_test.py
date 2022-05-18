import hnccorr.movie
from hnccorr import HNCcorr, Movie, HNCcorrConfig
from hnccorr.example import load_example_data
import tifffile
import numpy
import matplotlib.pyplot as plt


downsampling = 8  # dividing factor, 4 yields a 512x512 if the input is 2048x2048

filename = 's2a4d1_WF_1P_1x1_400mA_50Hz_func_500frames_4AP_1_MMStack.tif'

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
output = H.segmentations_to_list()  # Export list of cells (for Neurofinder),  Returns list[dict[tuple]]: List of cell coordinates.


for i in range(len(output)):
    print(i)
    zip(*output[i]['coordinates'])
    plt.scatter(*zip(*output[i]['coordinates']))

plt.ylim(0, 256)
plt.xlim(0, 256)
plt.xlabel('pixel')
plt.ylabel('pixel')
plt.title('Coordinates obtained using the cell extraction algorithm HNCcor')
plt.show()