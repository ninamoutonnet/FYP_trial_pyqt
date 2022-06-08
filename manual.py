

# Now that you have resized the whole stack, start the processing
# Read the image from the TIFF file as numpy array:
import numpy as np
from matplotlib import pyplot as plt
from mpmath import linspace, atan
from scipy.ndimage import gaussian_filter
from tifffile import tifffile
from skimage import io

'''
imfile = tifffile.imread('Downsampled_s1a2d1_WF_1P_1x1_300mA_100Hz_func_500frames_4AP_2_MMStack_Default.ome.tif')

# Take the mean, pixel per pixel of the whole image
#mean_img = imfile.mean(axis=0)
#plt.gray()

# Create the np array that will be the map
heat_map_np_array = np.copy(imfile)
heat_map_np_array = gaussian_filter(heat_map_np_array, sigma=3) #500, 1024, 1024
heat_map_np_array_mean = np.mean(heat_map_np_array, axis=0)# 1, 1024, 1024


gauss_highpass = imfile - heat_map_np_array
gauss_highpass_mean = imfile - heat_map_np_array_mean # GOOD ONE

minimum_value = np.min(gauss_highpass_mean)
gauss_highpass_mean_scaled = gauss_highpass_mean + abs(minimum_value)

mean_F0 = np.mean(gauss_highpass_mean_scaled, axis=0) #1, 1024, 1024
# value of darkness due to the microscope -> use the average of the first pixels in the top corner to do so
value_of_darkness = np.mean(gauss_highpass_mean_scaled[1:199, 1:5, 1:5])

gauss_highpass_mean_scaled_Variation = (gauss_highpass_mean_scaled - mean_F0)/mean_F0
gauss_highpass_mean_scaled_Variation_Dark = (gauss_highpass_mean_scaled - mean_F0 - value_of_darkness)/mean_F0

print('here')
#tifffile.imwrite('heat_map_np_array_mean.tif', heat_map_np_array_mean, photometric='minisblack')
#tifffile.imwrite('gauss_highpass.tif', gauss_highpass, photometric='minisblack')
#tifffile.imwrite('gauss_highpass_mean.tif', gauss_highpass_mean, photometric='minisblack')
#tifffile.imwrite('gauss_highpass_mean_scaled.tif', gauss_highpass_mean_scaled, photometric='minisblack')
#tifffile.imwrite('gauss_highpass_mean_scaled_Variation.tif', gauss_highpass_mean_scaled_Variation, photometric='minisblack')
tifffile.imwrite('gauss_highpass_mean_scaled_Variation_Dark.tif', gauss_highpass_mean_scaled_Variation_Dark, photometric='minisblack')

'''
#gauss_highpass_mean_scaled_Variation = io.imread('gauss_highpass_mean_scaled_Variation.tif')
gauss_highpass_mean_scaled_Variation = tifffile.imread('gauss_highpass_mean_scaled_Variation.tif', )
original_file = tifffile.imread('Downsampled_s1a2d1_WF_1P_1x1_300mA_100Hz_func_500frames_4AP_2_MMStack_Default.ome.tif')

'''# Check the organisation, the array stores the data as Z, Y, X!!
top_Left = gauss_highpass_mean_scaled_Variation[0, 0, 0]
top_Right = gauss_highpass_mean_scaled_Variation[0, 0, 1020]
bottom_Left = gauss_highpass_mean_scaled_Variation[0, 1022, 0]
bottom_Right = gauss_highpass_mean_scaled_Variation[0, 1022, 1022]

print('top_Left', top_Left)
print('top_Right', top_Right)
print('bottom_Left', bottom_Left)
print('bottom_Right', bottom_Right)'''


gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408 = gauss_highpass_mean_scaled_Variation[:, 408 , 642]
original_file_Trace_X_642_Y_408 = original_file[:, 408 , 642]

#'''
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2)

t = linspace(0, 499, 500)
# offset the processed one
gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408 = gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408 - (np.min(gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408))
axis[0].plot(gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408)
axis[0].set_title('Processed')
axis[0].grid()

original_file_Trace_X_642_Y_408 = original_file_Trace_X_642_Y_408 - np.min(original_file_Trace_X_642_Y_408)
axis[1].plot(original_file_Trace_X_642_Y_408)
axis[1].set_title('Original')
axis[1].grid()

plt.show()

'''

gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408 = gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408 - np.min(gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408)
processed_max = np.max(gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408)

original_file_Trace_X_642_Y_408 = original_file_Trace_X_642_Y_408 - np.min(original_file_Trace_X_642_Y_408)
original_max = np.max(original_file_Trace_X_642_Y_408)

percentage_process = gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408*100/processed_max
percentage_original = original_file_Trace_X_642_Y_408*100/original_max

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(3)
axis[0].plot(percentage_process)
axis[0].set_title('Processed')
axis[1].plot(percentage_original)
axis[1].set_title('Original')
axis[2].plot(percentage_original-percentage_process)
axis[2].set_title('Difference')
plt.show()
'''