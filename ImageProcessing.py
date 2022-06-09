from PIL import Image
from PIL import ImageSequence
from PIL import TiffImagePlugin
import numpy as np
import os
from matplotlib import pyplot as plt
from tifffile import tifffile
from scipy.ndimage import gaussian_filter



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

    # store the data in a temp array, use a high pass filter on all the 500 frames,
    # take a time average of that high-pass image (stored in temp_file_mean)
    temp_file = np.copy(imfile)
    temp_file = gaussian_filter(temp_file, sigma=3)  # 500, 1024, 1024
    temp_file_mean = np.mean(temp_file, axis=0)  # 1, 1024, 1024

    # remove the background (that mean high pass filter image) to the data
    gauss_highpass_mean = imfile - temp_file_mean

    # find the minimum value of gray, and scale the data such that it becomes 0
    minimum_value = np.min(gauss_highpass_mean)
    gauss_highpass_mean_scaled = gauss_highpass_mean - minimum_value

    # find the mean intensity of each pixel
    mean_F0 = np.mean(gauss_highpass_mean_scaled, axis=0)  # 1, 1024, 1024
    # Delta F / F
    gauss_highpass_mean_scaled_variation = (gauss_highpass_mean_scaled - mean_F0) / mean_F0

    gauss_highpass_mean_scaled_variation = gauss_highpass_mean_scaled_variation.astype(np.float32)
    OUTFILE = 'gauss_highpass_mean_scaled_variation.tif'
    with tifffile.Timer():
        tifffile.imwrite(OUTFILE, gauss_highpass_mean_scaled_variation, photometric='minisblack')
    return OUTFILE

