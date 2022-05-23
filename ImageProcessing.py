from PIL import Image
from PIL import ImageSequence
from PIL import TiffImagePlugin
import numpy as np
import os
from matplotlib import pyplot as plt
import tifffile
from scipy.ndimage.filters import gaussian_filter



def fluoMap(filename, downsample_factor):
    # # # # # # #
    #  source   # https://stackoverflow.com/questions/50702024/multi-page-tiff-resizing-python
    # # # # # # #

    # because the naming is automatic and you want a brand new file each time, ensure that no previous version

    file_name_no_path = os.path.basename(os.path.normpath(filename))
    name = 'Downsampled_' + file_name_no_path

    try:
        os.remove(name)
        os.remove('average_tif_resized.tif')
    except:
        pass

    pages = []
    imagehandler = Image.open(filename)
    for page in ImageSequence.Iterator(imagehandler):
        new_size = (int(page.size[0] / downsample_factor), int(page.size[1] / downsample_factor))
        page = page.resize(new_size)
        pages.append(page)
    with TiffImagePlugin.AppendingTiffWriter(name) as tf:
        for page in pages:
            page.save(tf)
            tf.newFrame()


    # Now that you have resized the whole stack, start the processing
    # Read the image from the TIFF file as numpy array:
    imfile = tifffile.imread(name)
    # Take the mean, pixel per pixel of the whole image
    mean_img = imfile.mean(axis=0)
    plt.gray()

    # Create the np array that will be the map
    heat_map_np_array = np.copy(imfile)

    # value of darkness due to the microscope -> use the average of the first pixels in the top corner to do so
    value_of_darkness = np.mean(imfile[1:20, 1:5, 1:5])

    for i in range(imfile.shape[0]):
        heat_map_np_array[i] = abs(imfile[i] - mean_img - value_of_darkness)
        heat_map_np_array[i] = gaussian_filter(heat_map_np_array[i], sigma=1)

    OUTFILE = 'average_tif_resized.tif'
    tifffile.imwrite(OUTFILE, heat_map_np_array, photometric='minisblack')

    return OUTFILE

