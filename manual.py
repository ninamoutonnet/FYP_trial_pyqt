import numpy as np
from matplotlib import pyplot as plt
from mpmath import linspace
from scipy.ndimage import gaussian_filter
from tifffile import tifffile

# read the downsampled .tiff file, store it in imfile as a numpy array
imfile = tifffile.imread('Downsampled_s1a2d1_WF_1P_1x1_300mA_100Hz_func_500frames_4AP_2_MMStack_Default.ome.tif')

# store the data in a temp array, use a high pass filter on all the 500 frames,
# take a time average of that high-pass image (stored in temp_file_mean)
temp_file = np.copy(imfile)
temp_file = gaussian_filter(temp_file, sigma=3) #500, 1024, 1024
temp_file_mean = np.mean(temp_file, axis=0)# 1, 1024, 1024
tifffile.imwrite('temp_file_mean.tif', temp_file_mean, photometric='minisblack')

# remove the background to the data, frame by frame
gauss_highpass = imfile - temp_file
tifffile.imwrite('gauss_highpass.tif', gauss_highpass, photometric='minisblack')

# remove the background (that mean high pass filter image) to the data
gauss_highpass_mean = imfile - temp_file_mean
tifffile.imwrite('gauss_highpass_mean.tif', gauss_highpass_mean, photometric='minisblack')

minimum_value = np.min(gauss_highpass_mean)
# find the minimum value of gray, and scale the data such that it becomes 0
gauss_highpass_mean_scaled = gauss_highpass_mean - minimum_value
tifffile.imwrite('gauss_highpass_mean_scaled.tif', gauss_highpass_mean_scaled, photometric='minisblack')

# find the mean intensity of each pixel
mean_F0 = np.mean(gauss_highpass_mean_scaled, axis=0) #1, 1024, 1024
# value of darkness due to the microscope -> use the average of the first pixels in the top corner to do so
value_of_darkness = np.mean(gauss_highpass_mean_scaled[1:199, 1:5, 1:5])
# Delta F / F
gauss_highpass_mean_scaled_Variation = (gauss_highpass_mean_scaled - mean_F0)/mean_F0
tifffile.imwrite('gauss_highpass_mean_scaled_Variation.tif', gauss_highpass_mean_scaled_Variation, photometric='minisblack')

# (F_pixel - F_0 - dark) / F_0
gauss_highpass_mean_scaled_Variation_Dark = (gauss_highpass_mean_scaled - mean_F0 - value_of_darkness)/mean_F0
tifffile.imwrite('gauss_highpass_mean_scaled_Variation_Dark.tif', gauss_highpass_mean_scaled_Variation_Dark, photometric='minisblack')

#gauss_highpass_mean_scaled_Variation = io.imread('gauss_highpass_mean_scaled_Variation.tif')
gauss_highpass_mean_scaled_Variation = tifffile.imread('gauss_highpass_mean_scaled_Variation.tif', )
original_file = tifffile.imread('Downsampled_s1a2d1_WF_1P_1x1_300mA_100Hz_func_500frames_4AP_2_MMStack_Default.ome.tif')

##################################################################
# Check the organisation, the array stores the data as Z, Y, X!! #
##################################################################

'''top_Left = gauss_highpass_mean_scaled_Variation[0, 0, 0]
top_Right = gauss_highpass_mean_scaled_Variation[0, 0, 1020]
bottom_Left = gauss_highpass_mean_scaled_Variation[0, 1022, 0]
bottom_Right = gauss_highpass_mean_scaled_Variation[0, 1022, 1022]

print('top_Left', top_Left)
print('top_Right', top_Right)
print('bottom_Left', bottom_Left)
print('bottom_Right', bottom_Right)'''


##################################################################
#                       Trace extraction                         #
##################################################################
'''gauss_highpass_mean_scaled_Variation_Trace_X_642_Y_408 = gauss_highpass_mean_scaled_Variation[:, 408 , 642]
original_file_Trace_X_642_Y_408 = original_file[:, 408 , 642]

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

##################################################################
#                       Comparison                               #
##################################################################
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