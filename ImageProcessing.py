from PIL import Image
from PIL import ImageSequence
from PIL import TiffImagePlugin
import numpy as np
import os
from matplotlib import pyplot as plt
import tifffile


def fluoMap(filename):
    # # # # # # #
    #  source   # https://stackoverflow.com/questions/50702024/multi-page-tiff-resizing-python
    # # # # # # #

    # because the naming is automatic and you want a brand new file each time, ensure that no previous version
    # named 'multipage_tif_resized.tif' exist.
    try:
        os.remove('multipage_tif_resized.tif')
    except:
        print("An exception occurred")

    INFILE = filename
    RESIZED_STACK = 'multipage_tif_resized.tif'
    OUTFILE2 = 'average_tif_resized.tif'

    downsampling_factor = 1
    pages = []
    imagehandler = Image.open(INFILE)
    for page in ImageSequence.Iterator(imagehandler):
        new_size = (int(page.size[0] / downsampling_factor), int(page.size[1] / downsampling_factor))
        page = page.resize(new_size)
        pages.append(page)
    #print('Writing multipage TIF: ', pages[0])
    with TiffImagePlugin.AppendingTiffWriter(RESIZED_STACK) as tf:
        for page in pages:
            page.save(tf)
            tf.newFrame()

    '''infile = tifffile.imread(INFILE)
    print('shape is: ', infile.shape)
    infile.resize(200, 256, 256)
    plt.gray()
    plt.imshow(infile)
    plt.show()
    print(infile[0, 0, 0])'''

    # Now that you have resized the whole stack, start the processing
    # Read the image from the TIFF file as numpy array:
    imfile = tifffile.imread(RESIZED_STACK)
    # image_array = imfile.astype('uint8')
    # print('shape of 8 bit is: ', image_array.shape)
    # plt.gray()
    # plt.imshow(image_array[0])
    # plt.show()

    # print(imfile.dtype, image_array.dtype)
    # print('max: ', np.amax(imfile), ' ', np.amax(image_array))
    # print('min: ', np.amin(imfile), ' ', np.amin(image_array))

    # Take the mean, pixel per pixel of the whole image
    OUTFILE2_NP = imfile.mean(axis=0)
    plt.gray()
    #plt.imshow(OUTFILE2_NP)
    #plt.show()

    # Create the np array that will be the map
    heat_map_np_array = np.copy(imfile)
    # print(heat_map_np_array[1])
    # print(OUTFILE2_NP)
    # print((heat_map_np_array[1]-OUTFILE2_NP) / OUTFILE2_NP)
    # print(np.amax((heat_map_np_array[1]-OUTFILE2_NP) / OUTFILE2_NP))
    # print(np.amin((heat_map_np_array[1] - OUTFILE2_NP) / OUTFILE2_NP))

    for i in range(imfile.shape[0]):
        temp = (imfile[i] - OUTFILE2_NP) / OUTFILE2_NP
        min = np.amin(temp)
        max = np.amax(temp)
        scaling_factor = (max - min) / 65535
        print(min, '   ', max, '   ', scaling_factor, np.amin((temp - min)) / scaling_factor, ' ',
              np.amax((temp - min)) / scaling_factor)
        # heat_map_np_array[i] = (temp-min)/scaling_factor
        heat_map_np_array[i] = abs(imfile[i] - OUTFILE2_NP)

    tifffile.imwrite(OUTFILE2, heat_map_np_array, photometric='minisblack')
    # print('size of fluo output: ', heat_map_np_array.shape)

    return OUTFILE2


def colourMap(filename):
    try:
        os.remove('multipage_tif_resized_COLOUR_MAP.tif')
    except:
        print("An exception occurred")

    OUTFILE = "multipage_tif_resized_COLOUR_MAP.tif"
    imfile = tifffile.imread('average_tif_resized.tif')

    return