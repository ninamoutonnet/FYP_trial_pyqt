import cv2
from PIL import Image
from PIL import ImageSequence
from PIL import TiffImagePlugin
import numpy as np
import os
# Importing the statistics module
import statistics
from matplotlib import pyplot as plt
import tifffile

# # # # # # #
#  source   #  https://stackoverflow.com/questions/50702024/multi-page-tiff-resizing-python
# # # # # # #


def fluoMap(filename):
    # # # # # # #
    #  source   # https://stackoverflow.com/questions/50702024/multi-page-tiff-resizing-python
    # # # # # # #

    os.remove('multipage_tif_resized.tif')

    INFILE = filename
    RESIZED_STACK = 'multipage_tif_resized.tif'
    OUTFILE2 = 'average_tif_resized.tif'


    # print('Resizing TIF pages')
    '''pages = []
    imagehandler = Image.open(INFILE)
    for page in ImageSequence.Iterator(imagehandler):
        new_size = (int(page.size[0]/8), int(page.size[1]/8))
        page = page.resize(new_size)
        pages.append(page)
    print('Writing multipage TIF: ', pages[0])
    with TiffImagePlugin.AppendingTiffWriter(RESIZED_STACK) as tf:
        for page in pages:
            page.save(tf)
            tf.newFrame()'''

    infile = tifffile.imread(INFILE)
    print('shape is: ', infile.shape)
    infile.resize(int(infile.shape[1]), int(infile.shape[2]))
    plt.gray()
    plt.imshow(infile)
    plt.show()
    print(infile[0, 0, 0])

    # Now that you have resized the whole stack, start the processing
    # Read the image from the TIFF file as numpy array:
    imfile = tifffile.imread(RESIZED_STACK)
    print(imfile[0,0,0])

    # Take the mean, pixel per pixel of the whole image
    OUTFILE2_NP = imfile.mean(axis=0)
    print(OUTFILE2_NP[0,0])
    #plt.gray()
    #plt.imshow(OUTFILE2_NP)
    #plt.show()

    # Create the np array that will be the map
    heat_map_np_array = imfile
    for i in range(imfile.shape[0]):
        heat_map_np_array[i] = (heat_map_np_array[i]/OUTFILE2_NP)*255

    tifffile.imwrite(OUTFILE2, heat_map_np_array, photometric='minisblack')
    return OUTFILE2

    ''''# Read the tiff stack
    # this reads the tiff file as a gray scale image
    ret, images = cv2.imreadmulti(filename, [], cv2.IMREAD_GRAYSCALE)
    imgArr = images[0]  # note 0 based indexing, this is the 1st image of the stiff

    # compress the image to a 256 by 256 image
    img = Image.fromarray(imgArr)
    new_image = img.resize((256, 256))
    # convert it into an array
    np_array_low_resolution_img = numpy.array(new_image)'''



