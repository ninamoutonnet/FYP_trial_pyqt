from PIL import Image
from PIL import ImageSequence
from PIL import TiffImagePlugin
import numpy as np
import os
from matplotlib import pyplot as plt
import tifffile



def fluoMap(filename, downsample_factor):
    # # # # # # #
    #  source   # https://stackoverflow.com/questions/50702024/multi-page-tiff-resizing-python
    # # # # # # #

    # because the naming is automatic and you want a brand new file each time, ensure that no previous version
    # named 'multipage_tif_resized.tif' exist.
    try:
        os.remove('multipage_tif_resized.tif')
        os.remove('average_tif_resized.tif')
    except:
        pass

    INFILE = filename
    RESIZED_STACK = 'multipage_tif_resized.tif'
    OUTFILE = 'average_tif_resized.tif'

    pages = []
    imagehandler = Image.open(INFILE)
    for page in ImageSequence.Iterator(imagehandler):
        new_size = (int(page.size[0] / downsample_factor), int(page.size[1] / downsample_factor))
        page = page.resize(new_size)
        pages.append(page)
    with TiffImagePlugin.AppendingTiffWriter(RESIZED_STACK) as tf:
        for page in pages:
            page.save(tf)
            tf.newFrame()


    # Now that you have resized the whole stack, start the processing
    # Read the image from the TIFF file as numpy array:
    imfile = tifffile.imread(RESIZED_STACK)
    # Take the mean, pixel per pixel of the whole image
    mean_img = imfile.mean(axis=0)
    plt.gray()

    # Create the np array that will be the map
    heat_map_np_array = np.copy(imfile)

    # value of darkness due to the microscope -> use the average of the first pixels in the top corner to do so
    value_of_darkness = np.mean(imfile[1:5,1:5, 1:20])

    for i in range(imfile.shape[0]):
        heat_map_np_array[i] = abs(imfile[i] - mean_img - value_of_darkness)

    tifffile.imwrite(OUTFILE, heat_map_np_array, photometric='minisblack')

    return OUTFILE

