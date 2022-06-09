from hnccorr import HNCcorr, Movie, HNCcorrConfig
import tifffile
import numpy
import matplotlib.pyplot as plt
import cv2


downsampling = 8  # dividing factor, 4 yields a 512x512 if the input is 2048x2048

filename = 's2a4d1_WF_1P_1x1_400mA_50Hz_func_500frames_4AP_1_MMStack_Default.tif'

with tifffile.Timer():
    stack = tifffile.imread(filename)[:, ::downsampling, ::downsampling].copy()

with tifffile.Timer():
    with tifffile.TiffFile(filename) as tif:
        page = tif.pages[0]
        shape = len(tif.pages), page.imagelength // downsampling, page.imagewidth // downsampling
        stack = numpy.empty(shape, page.dtype)
        for i, page in enumerate(tif.pages):
            # better use interpolation instead:
            stack[i] = cv2.resize(page.asarray(), dsize=(shape[2], shape[1]), interpolation=cv2.INTER_LINEAR)

print(stack.dtype)


movie = Movie('exp1', stack)
with tifffile.Timer():
    config = HNCcorrConfig(postprocessor_min_cell_size = 5, postprocessor_preferred_cell_size = 10,postprocessor_max_cell_size = 100, patch_size = 31)
    H = HNCcorr.from_config(config)  # Initialize HNCcorr with default configuration
    H.segment(movie)    # perform the decomposition algorithm

H.segmentations  # List of identified cells
output = H.segmentations_to_list()  # Export list of cells (for Neurofinder),  Returns list[dict[tuple]]: List of cell coordinates.


for i in range(len(output)):
    print(i)
    zip(*output[i]['coordinates'])
    plt.scatter(*zip(*output[i]['coordinates']))

plt.ylim(0, 256)
plt.xlim(0, 256)
plt.grid()
plt.xlabel('pixel value (x axis)')
plt.ylabel('pixel value (y axis)')
plt.title('Location of the firing neurons extracted using the cell extraction algorithm HNCcorr')
plt.savefig('HNCcorr_results.pdf', dpi=1200)
plt.show()
