import sys
from PyQt5 import QtCore
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from MultiPageTIFFViewerQt import MultiPageTIFFViewerQt
import pyqtgraph as pg
from ImageProcessing import fluoMap
import tifffile
import os
from PIL import Image
from PIL import ImageSequence
from PIL import TiffImagePlugin
import numpy as np
import CNMFE
import PCA

class PixelTemporalVariation(qtw.QWidget):

    def __init__(self, x, y, filename, acquisition_rate):
        super().__init__()
        #  start UI code
        # QLabel
        label = qtw.QLabel('Pixel Variation 1D', self, margin=10)

        # Read the image from the TIFF file as numpy array:
        imfile = tifffile.imread(filename)
        # value of the darkness due to the microscope is the average of the first 5x5 pixel
        # square at the top of the image over the whole stack

        value_of_darkness = np.mean(imfile[:, 1:5, 1:5])
        extracted_temporal_trace = imfile[:, x, y]
        f_0 = np.mean(imfile)
        extracted_temporal_trace = (extracted_temporal_trace - f_0)/ (f_0 - value_of_darkness)

        # using the acquisition rate, tranform the x axis from frame numbet to seconds
        timescale = np.arange(0, imfile.shape[0]/acquisition_rate, 1/acquisition_rate)

        graphWidget = pg.PlotWidget()
        pen = pg.mkPen(color=(255, 0, 0), width=1)
        graphWidget.setBackground('w')
        graphWidget.plot(timescale, extracted_temporal_trace, pen=pen)
        graphWidget.showGrid(x=True, y=True)
        graphWidget.setTitle("Temporal Trace of ("+str(x)+","+str(y)+f") of {filename}", size="30pt")
        styles = {'color': 'b', 'font-size': '20px'}
        graphWidget.setLabel('left', 'delta F / F', **styles)
        graphWidget.setLabel('bottom', 'time, s', **styles)

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(label)
        layout.addWidget(graphWidget)

        #  end main UI code - Display the UI
        self.show()


class FluorescenceIntensityMap(qtw.QWidget):

    def __init__(self, filename, downsample_factor, acquisition_rate):

        '''Main Window constructor'''
        super().__init__()
        #  get the filname/path of the original tiff stack to use when looking at individual pixel intensity
        self.filename = filename
        self.acquisition_rate = acquisition_rate

        # strip the name of the file from the path
        self.filename_no_path = os.path.basename(os.path.normpath(self.filename))
        self.filename_downsampled = 'Downsampled_' + self.filename_no_path


        #  start UI code

        # give the tiff file to the intensity function
        self.fluo_output = fluoMap(filename, downsample_factor)
        self.stackViewer = MultiPageTIFFViewerQt()
        self.stackViewer.loadImageStackFromFile(self.fluo_output)

        im = Image.open(self.fluo_output)
        self.width_input_file, self.height_input_file = im.size

        # Create a button to see 1D intensity variation
        # QPushButton
        button = qtw.QPushButton(
            "Pixel temporal traces",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.stackViewer)
        layout.addWidget(button)

        #  end main UI code - Display the UI
        self.show()

        #  If the button is pressed, open a new window
        button.clicked.connect(self.WindowIntensityVariation)


    def getPosition(self, event):
        # get the viewer size
        viewer_size = (self.stackViewer.viewer.size().width(), self.stackViewer.viewer.size().height())
        # print('viewer size ', viewer_size)
        # get the image size
        image_size = (self.width_input_file, self.height_input_file)
        # print('image size ', image_size)

        x = event.pos().x()
        y = event.pos().y()

        # 3 possible scenarios; assume that the input is SQUARE
        if viewer_size[0] > viewer_size[1]: # the width of the viewer is bigger than the height
            # print('Width > height')
            scaling_factor = image_size[1]/viewer_size[1]
            x_offset = (viewer_size[0] - viewer_size[1])/2
            y_coordinate = y*scaling_factor
            x_coordinate = (x-x_offset)*scaling_factor

        elif viewer_size[0] < viewer_size[1]:# the width of the viewer is smaller than the height
            # print('height > width')
            scaling_factor = image_size[0] / viewer_size[0]
            y_offset = (viewer_size[1] - viewer_size[0]) / 2
            y_coordinate = (y-y_offset) * scaling_factor
            x_coordinate = x * scaling_factor

        else:  # the width of the viewer  =  height
            # print('Width = height')
            scaling_factor = image_size[0] / viewer_size[0]
            y_coordinate = y * scaling_factor
            x_coordinate = x * scaling_factor

        self.w = PixelTemporalVariation(round(x_coordinate), round(y_coordinate), self.filename_downsampled, self.acquisition_rate)
        self.w.show()

    def WindowIntensityVariation(self):
        # Add event when the buttons are pressed -> get the position of interest
        self.stackViewer.viewer.mousePressEvent = self.getPosition


class CNMFE_GUI(qtw.QWidget):

    def __init__(self, filename):

        '''Main Window constructor'''
        super().__init__()
        #  get the filname/path of the original tiff stack to use when looking at individual pixel intensity
        self.filename = filename

        #  start UI code

        # Create the text boxes to enter the parameters, use a QFormLayout widget
        variable_layout = qtw.QFormLayout()
        self.setLayout(variable_layout)
        self.setWindowTitle('CNMF-E parameters')

        qtw.QToolTip.setFont(qtg.QFont('arial',20))

        # create the text widgets for the entry of parameters
        self.lineEdit_average_cell_diameter = qtw.QLineEdit()
        self.lineEdit_average_cell_diameter.setPlaceholderText('7') #show the default value
        label_average_cell_diameter = qtw.QLabel('Average cell diameter')
        label_average_cell_diameter.setToolTip('The average cell diameter of a representative cell in pixels,  <b> default = 7 <b>')
        variable_layout.addRow(label_average_cell_diameter, self.lineEdit_average_cell_diameter)

        self.lineEdit_min_pixel_correlation = qtw.QLineEdit()
        self.lineEdit_min_pixel_correlation.setPlaceholderText('0.8')
        label_min_pixel_correlation = qtw.QLabel('Minimum pixel correlation')
        label_min_pixel_correlation.setToolTip('The minimum correlation of a pixel with its immediate neighbors when searching for new cell centers,  <b> default = 0.8 <b>')
        variable_layout.addRow(label_min_pixel_correlation, self.lineEdit_min_pixel_correlation)

        self.lineEdit_min_peak_to_noise_ratio = qtw.QLineEdit()
        self.lineEdit_min_peak_to_noise_ratio.setPlaceholderText('10.0')
        label_min_peak_to_noise_ratio = qtw.QLabel('min_peak_to_noise_ratio')
        label_min_peak_to_noise_ratio.setToolTip('The minimum peak-to-noise ratio of a pixel when searching for new cell centers,  <b> default = 10 <b>')
        variable_layout.addRow(label_min_peak_to_noise_ratio, self.lineEdit_min_peak_to_noise_ratio)

        self.lineEdit_gaussian_kernel_size = qtw.QLineEdit()
        self.lineEdit_gaussian_kernel_size.setPlaceholderText('0')
        label_gaussian_kernel_size = qtw.QLabel('Gaussian kernel size')
        label_gaussian_kernel_size.setToolTip('The width in pixels of the Gaussian kernel used for spatial filtering of the movie before cell initialization (automatically estimated when the value provided is smaller than 3),  <b> default = 0 <b>')
        variable_layout.addRow(label_gaussian_kernel_size, self.lineEdit_gaussian_kernel_size)

        self.lineEdit_closing_kernel_size = qtw.QLineEdit()
        self.lineEdit_closing_kernel_size.setPlaceholderText('0')
        label_closing_kernel_size = qtw.QLabel('Closing kernel size')
        label_closing_kernel_size.setToolTip('The size in pixels of the morphological closing kernel used for removing small disconnected components and connecting small cracks within individual cell footprints (automatically estimated when the value provided is smaller than 3),  <b> default = 0 <b>')
        variable_layout.addRow(label_closing_kernel_size, self.lineEdit_closing_kernel_size)

        self.lineEdit_background_downsampling_factor = qtw.QLineEdit()
        self.lineEdit_background_downsampling_factor.setPlaceholderText('2')
        label_background_downsampling_factor = qtw.QLabel('Background downsampling factor')
        label_background_downsampling_factor.setToolTip('The spatial downsampling factor to use when estimating the background activity,  <b> default = 2 <b>')
        variable_layout.addRow(label_background_downsampling_factor, self.lineEdit_background_downsampling_factor)

        self.lineEdit_ring_size_factor = qtw.QLineEdit()
        self.lineEdit_ring_size_factor.setPlaceholderText('1.4')
        label_ring_size_factor = qtw.QLabel('Ring size factor')
        label_ring_size_factor.setToolTip('The multiple of the average cell diameter to use for computing the radius of the ring model used for estimating the background activity,  <b> default = 1.4 <b>')
        variable_layout.addRow(label_ring_size_factor, self.lineEdit_ring_size_factor)

        self.lineEdit_merge_threshold = qtw.QLineEdit()
        self.lineEdit_merge_threshold.setPlaceholderText('0.7')
        label_merge_threshold = qtw.QLabel('Merge threshold')
        label_merge_threshold.setToolTip('The temporal correlation threshold for merging cells that are spatially close,  <b> default = 0.7 <b>')
        variable_layout.addRow(label_merge_threshold, self.lineEdit_merge_threshold)

        self.lineEdit_num_threads = qtw.QLineEdit()
        self.lineEdit_num_threads.setPlaceholderText('4')
        label_num_threads = qtw.QLabel('Number of threads')
        label_num_threads.setToolTip('The number of threads to use for processing	,  <b> default = 4 <b>')
        variable_layout.addRow(label_num_threads, self.lineEdit_num_threads)

        self.lineEdit_processing_mode = qtw.QLineEdit()
        self.lineEdit_processing_mode.setPlaceholderText('2')
        label_processing_mode = qtw.QLabel('Processing mode')
        label_processing_mode.setToolTip('the processing mode to use to run CNMF-E (0: all in memory, 1: sequential patches, 2: parallel patches). All in memory: processes the entire field of view at once. Sequential patches: breaks the field of view into overlapping patches and processes them one at a time using the specified number of threads where parallelization is possible. Parallel patches: breaks the field of view into overlapping patches and processes them in parallel using a single thread for each.,  <b> default = 2 <b>')
        variable_layout.addRow(label_processing_mode, self.lineEdit_processing_mode)

        self.lineEdit_patch_size = qtw.QLineEdit()
        self.lineEdit_patch_size.setPlaceholderText('80')
        label_patch_size = qtw.QLabel('Patch size')
        label_patch_size.setToolTip('The side length of an individual square patch of the field of view in pixels,  <b> default = 80 <b>')
        variable_layout.addRow(label_patch_size, self.lineEdit_patch_size)

        self.lineEdit_patch_overlap = qtw.QLineEdit()
        self.lineEdit_patch_overlap.setPlaceholderText('20')
        label_patch_overlap = qtw.QLabel('Patch overlap')
        label_patch_overlap.setToolTip('The amount of overlap between adjacent patches in pixels,  <b> default = 20 <b>')
        variable_layout.addRow(label_patch_overlap, self.lineEdit_patch_overlap)

        self.lineEdit_deconvolve = qtw.QLineEdit()
        self.lineEdit_deconvolve.setPlaceholderText('0')
        label_deconvolve = qtw.QLabel('Deconvolve')
        label_deconvolve.setToolTip('Specifies whether to deconvolve the final temporal traces (0: return raw traces, 1: return deconvolved traces),  <b> default = 0 <b>')
        variable_layout.addRow(label_deconvolve, self.lineEdit_deconvolve)

        self.lineEdit_output_units = qtw.QLineEdit()
        self.lineEdit_output_units.setPlaceholderText('1')
        label_output_units = qtw.QLabel('Output units')
        label_output_units.setToolTip('The units of the output temporal traces (0: dF, 1: dF over noise). dF: temporal traces on the same scale of pixel intensity as the original movie. dF is calculated as the average fluorescence activity of all pixels in a cell, scaled so that each spatial footprint has a magnitude of 1. dF over noise: temporal traces divided by their respective estimated noise level. This can be interpreted similarly to a z-score, with the added benefit that the noise is a more robust measure of the variance in a temporal trace compared to the standard deviation.,  <b> default = 1 <b>')
        variable_layout.addRow(label_output_units, self.lineEdit_output_units)

        self.lineEdit_output_filetype = qtw.QLineEdit()
        self.lineEdit_output_filetype.setPlaceholderText('0')
        label_output_filetype = qtw.QLabel('Output filetype')
        label_output_filetype.setToolTip('The file types into which the output will be saved (0: footprints saved to a tiff file and traces saved to a csv file, 1: output saved to a h5 file under the keys footprints and traces),  <b> default = 0 <b>')
        variable_layout.addRow(label_output_filetype, self.lineEdit_output_filetype)

        self.lineEdit_verbose = qtw.QLineEdit()
        self.lineEdit_verbose.setPlaceholderText('1')
        label_verbose = qtw.QLabel('Verbose')
        label_verbose.setToolTip('To enable and disable verbose mode. When enabled, progress is displayed in the console. (0: disabled, 1: enabled),  <b> default = 0 <b>')
        variable_layout.addRow(label_verbose, self.lineEdit_verbose)

        # QPushButton
        button = qtw.QPushButton(
            "Run CNMFE",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )
        variable_layout.addWidget(button)

        #  end main UI code - Display the UI
        self.show()

        #  If the button is pressed, open a new window
        button.clicked.connect(self.get_cnmfe_parameters)

    def get_cnmfe_parameters(self):

        # first of all, extract the information from the text boxes
        average_cell_diameter = self.lineEdit_average_cell_diameter.text()
        min_pixel_correlation = self.lineEdit_min_pixel_correlation.text()
        min_peak_to_noise_ratio = self.lineEdit_min_peak_to_noise_ratio.text()
        gaussian_kernel_size = self.lineEdit_gaussian_kernel_size.text()
        closing_kernel_size = self.lineEdit_closing_kernel_size.text()
        background_downsampling_factor = self.lineEdit_background_downsampling_factor.text()
        ring_size_factor = self.lineEdit_ring_size_factor.text()
        merge_threshold = self.lineEdit_merge_threshold.text()
        num_threads = self.lineEdit_num_threads.text()
        processing_mode = self.lineEdit_processing_mode.text()
        patch_size = self.lineEdit_patch_size.text()
        patch_overlap = self.lineEdit_patch_overlap.text()
        deconvolve = self.lineEdit_deconvolve.text()
        output_units = self.lineEdit_output_units.text()
        output_filetype = self.lineEdit_output_filetype.text()
        verbose = self.lineEdit_verbose.text()

        # in case the field is not filled, set default values
        if average_cell_diameter == '':
            # integer
            average_cell_diameter = 7
        else:
            average_cell_diameter = int(average_cell_diameter)

        if min_pixel_correlation == '':
            # float
            min_pixel_correlation = 0.8
        else:
            min_pixel_correlation = float(min_pixel_correlation)

        if min_peak_to_noise_ratio == '':
            # float
            min_peak_to_noise_ratio = 10.0
        else:
            min_peak_to_noise_ratio = float(min_peak_to_noise_ratio)

        if gaussian_kernel_size == '':
            # int
            gaussian_kernel_size = 0
        else:
            gaussian_kernel_size = int(gaussian_kernel_size)

        if closing_kernel_size == '':
            # int
            closing_kernel_size = 0
        else:
            closing_kernel_size = int(closing_kernel_size)

        if background_downsampling_factor == '':
            # int
            background_downsampling_factor = 2
        else:
            background_downsampling_factor = int(background_downsampling_factor)

        if ring_size_factor == '':
            # float
            ring_size_factor = 1.4
        else:
            ring_size_factor = float(ring_size_factor)

        if merge_threshold == '':
            # float
            merge_threshold = 0.7
        else:
            merge_threshold = float(merge_threshold)

        if num_threads == '':
            num_threads = 4
        else:
            num_threads = int(num_threads)

        if processing_mode == '':
            processing_mode = 2
        else:
            processing_mode = int(processing_mode)

        if patch_size == '':
            print('no values')
            patch_size = 80
        else:
            print(' values')

            patch_size = int(patch_size)

        if patch_overlap == '':
            patch_overlap = 2
        else:
            patch_overlap = int(patch_overlap)

        if deconvolve == '':
            deconvolve = 0
        else:
            deconvolve = int(deconvolve)

        if output_units == '':
            output_units = 1
        else:
            output_units = int(output_units)

        if output_filetype == '':
            output_filetype = 0
        else:
            output_filetype = int(output_filetype)

        if verbose == '':
            verbose = 1
        else:
            verbose = int(verbose)

        # second, check the validity. If the input is ok, generate the fluorescence variation map
        print(f'OK - RUNNING CNMF-E')
        #  create a new CNMFE object, give it the resized filename as input
        cnmfe_object = CNMFE.CNMFE_class(self.filename,
                          average_cell_diameter,
                          min_pixel_correlation,
                          min_peak_to_noise_ratio,
                          gaussian_kernel_size,
                          closing_kernel_size,
                          background_downsampling_factor,
                          ring_size_factor,
                          merge_threshold,
                          num_threads,
                          processing_mode,
                          patch_size,
                          patch_overlap,
                          deconvolve,
                          output_units,
                          output_filetype,
                          verbose)

        cnmfe_object.plot_summary()


class PCA_GUI(qtw.QWidget):

    def __init__(self, filename):
        '''Main Window constructor'''
        super().__init__()
        #  get the filname/path of the original tiff stack to use when looking at individual pixel intensity
        self.filename = filename # this is the downsampled tiff file
        print(filename)
        # perform PCA
        pca_object = PCA.PCA_class(self.filename)



class MainWindow(qtw.QWidget):

    def __init__(self):
        '''Main Window constructor'''
        super().__init__()

        # main UI code goes here

        # Create an image stack viewer widget.
        self.stackViewer = MultiPageTIFFViewerQt()

        # As this next command has no argument, the files will pop up and the user will be asked to get the tiff stack to open
        self.stackViewer.loadImageStackFromFile()

        # the file path is stored in filename
        self.filename = self.stackViewer.openFileName

        # Create a button to see the varying intensity: QPushButton
        button = qtw.QPushButton(
            "Fluorescence variation Map",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        # Create a button to see 1D intensity variation: QPushButton
        button_pixel_intensity_var = qtw.QPushButton(
            "Pixel temporal intensity variation",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        # Create the labels at the top containing the size of tiff stack, it's title and it's path
        l1 = qtw.QLabel()
        l2 = qtw.QLabel()
        l3 = qtw.QLabel()
        l4 = qtw.QLabel()

        l1.setText("Calcium Imaging Analysis")
        l1.setAlignment(QtCore.Qt.AlignCenter)
        l1.setFont(qtg.QFont('Arial', 15, weight=qtg.QFont.Bold))

        byte_size_input_file = os.path.getsize(self.filename)
        if byte_size_input_file >= 10**9:
            l2.setText(f'Size: {byte_size_input_file/(10**9)} GB')
        elif byte_size_input_file >= 10**6:
            l2.setText(f'Size: {os.path.getsize(self.filename)/(10**6)} MB')
        elif byte_size_input_file >= 10 ** 3:
            l2.setText(f'Size: {os.path.getsize(self.filename) / (10 ** 3)} KB')
        else:
            l2.setText(f'Size: {os.path.getsize(self.filename)} Bytes')

        im = Image.open(self.filename)
        self.width_input_file, self.height_input_file = im.size
        l3.setText(f"Frame pixel size: {self.width_input_file}x{self.height_input_file}")

        # strip the name of the file from the path
        file_name = os.path.basename(os.path.normpath(self.filename))
        l4.setText('File name: ' + file_name)
        self.filename_no_path = file_name

        # Create a button for each cell sorting algorithm
        button_CNMFE = qtw.QPushButton(
            "CNMFE",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        button_PCA = qtw.QPushButton(
            "PCA",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        button_ORPCA = qtw.QPushButton(
            "ORPCA",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        button_Manual = qtw.QPushButton(
            "Manual",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        # create the text widgets for the entry of the downsampling factor and sampling rate
        self.downsample_value_widget = qtw.QLineEdit()
        # check that the downsample is an integer between 1 and the current pixel size
        self.downsample_value_widget.setValidator(qtg.QIntValidator(1, int(self.width_input_file)))

        self.sampling_rate_value_widget = qtw.QLineEdit()
        # check that the sampling rate is an integer between 1 and 1000 HZ
        self.sampling_rate_value_widget.setValidator(qtg.QIntValidator(1, 1000))

        # create the 'base' widget on which all the other widgets will be
        central_QWidget = qtw.QWidget()

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(central_QWidget)

        layout_2 = qtw.QHBoxLayout()
        central_QWidget.setLayout(layout_2)
        layout_2.addWidget(self.stackViewer)
        right_layout = qtw.QVBoxLayout()
        layout_2.addLayout(right_layout)

        right_layout.addWidget(l1)

        layout3 = qtw.QHBoxLayout()
        metadata_widget = qtw.QWidget()
        metadata_widget.setLayout(layout3)
        # titles
        title_layout = qtw.QVBoxLayout()
        layout3.addLayout(title_layout)
        title_layout.addWidget(l2)
        title_layout.addWidget(l3)
        title_layout.addWidget(l4)
        # Variables
        variable_layout = qtw.QFormLayout()
        layout3.addLayout(variable_layout)
        variable_layout.addRow('Downsampling value', self.downsample_value_widget)
        variable_layout.addRow('Acquisition rate in Hz', self.sampling_rate_value_widget)
        right_layout.addWidget(metadata_widget)

        simple_processing_widget = qtw.QGroupBox('Quick overview')
        simple_processing_widget.setFont(qtg.QFont('Arial', 12, weight=qtg.QFont.Bold))
        right_layout.addWidget(simple_processing_widget)
        simple_processing_layout = qtw.QHBoxLayout()
        simple_processing_widget.setLayout(simple_processing_layout)
        simple_processing_layout.addWidget(button_pixel_intensity_var)
        simple_processing_layout.addWidget(button)

        complex_processing_widget = qtw.QGroupBox('Cell segmentation')
        complex_processing_widget.setFont(qtg.QFont('Arial', 12, weight=qtg.QFont.Bold))
        right_layout.addWidget(complex_processing_widget)
        complex_processing_layout = qtw.QVBoxLayout()
        complex_processing_widget.setLayout(complex_processing_layout)
        complex_processing_layout.addWidget(button_Manual)
        complex_processing_layout.addWidget(button_PCA)
        complex_processing_layout.addWidget(button_ORPCA)
        complex_processing_layout.addWidget(button_CNMFE)

        complex_processing_widget.setSizePolicy(
            qtw.QSizePolicy.Preferred,
            qtw.QSizePolicy.Expanding
        )

        button_Manual.setSizePolicy(
            qtw.QSizePolicy.Preferred,
            qtw.QSizePolicy.Preferred
        )

        button_PCA.setSizePolicy(
            qtw.QSizePolicy.Preferred,
            qtw.QSizePolicy.Preferred
        )

        button_ORPCA.setSizePolicy(
            qtw.QSizePolicy.Preferred,
            qtw.QSizePolicy.Preferred
        )

        button_CNMFE.setSizePolicy(
            qtw.QSizePolicy.Preferred,
            qtw.QSizePolicy.Preferred
        )

        #  end main UI code - Display the UI
        self.show()

        #  If the button for fluorescence is pressed, open a new window
        button.clicked.connect(self.windowFluo)
        #  If the button is pressed, open a new window
        button_pixel_intensity_var.clicked.connect(self.WindowIntensityVariation)
        #  If the for CNMFE is pressed, open a new window
        #        button_CNMFE.clicked.connect(self.CNMFE_instance)
        button_CNMFE.clicked.connect(self.cnmfe_trial)

        button_PCA.clicked.connect(self.pca_trial)



    def pca_trial(self):
        # extract the info from the main GUI window
        downsample_factor = self.get_downsampling_value()
        # downsample the file and store it automatically as 'multipage_tif_resized.tif'
        filename_downsampled = self.downsampling_tiff_stack(self.filename, downsample_factor)
        # use that downsampled file to call the PCA/SVD
        self.pca_window = PCA_GUI(filename_downsampled)
        return

    def cnmfe_trial(self):
        # extract the info from the main GUI window
        downsample_factor = self.get_downsampling_value()
        # downsample the file and store it automatically as 'multipage_tif_resized.tif'
        filename_downsampled = self.downsampling_tiff_stack(self.filename, downsample_factor)
        # use that downsampled file to call the CNMFE
        self.cnmf_window = CNMFE_GUI(filename_downsampled)

        return

    def downsampling_tiff_stack(self, filename, downsample_factor):
        # delete any existing version of the file you are about to over write
        name = 'Downsampled_' + self.filename_no_path

        try:
            os.remove(name)
        except:
            pass

        # reduce the size of the file
        pages = []
        imagehandler = Image.open(self.filename)
        for page in ImageSequence.Iterator(imagehandler):
            new_size = (int(page.size[0] / downsample_factor), int(page.size[1] / downsample_factor))
            page = page.resize(new_size)
            pages.append(page)
        with TiffImagePlugin.AppendingTiffWriter(name) as tf:
            for page in pages:
                page.save(tf)
                tf.newFrame()

        return name

    def get_downsampling_value(self):
        # first of all, extract the information from the text box
        downsample = self.downsample_value_widget.text()
        # in case the field is not filled, the default value if 1
        if downsample == '':
            downsample = 1
        else:
            downsample = int(downsample)

        # second, check the validity. If the input is ok, generate the fluorescence variation map
        if downsample > int(self.width_input_file) or downsample < 1:
            print(f'The downsampling factor should be between 1 and {self.width_input_file}')
            return None
        # check that the division of the width AND heigth by the downsampling factor is an integer.
        if (int(self.width_input_file) % int(downsample) != 0) or (int(self.height_input_file) % int(downsample) != 0):
            print(f'The width and height of the tiff file should be a multiple of the downsampling factor')
            return None
        else:
            return downsample

    def get_acquisition_rate(self):
        acquisition_rate = self.sampling_rate_value_widget.text()
        if acquisition_rate == '':
            acquisition_rate = 50  # in case the field is not filled, the default value if 50 Hz
        else:
            acquisition_rate = int(acquisition_rate)

        return acquisition_rate

    def WindowIntensityVariation(self):
        # When the mouse press occurs on the stackviewer, execute the getPosition function,
        # which gets the position and plots the variation in intensity
        self.stackViewer.viewer.mousePressEvent = self.getPosition

    def windowFluo(self):
        downsample = self.get_downsampling_value()
        acquisition_rate = self.get_acquisition_rate()
        if downsample is not None:
            self.w = FluorescenceIntensityMap(self.filename, downsample, acquisition_rate)

    def getPosition(self, event):
        # get the viewer size
        viewer_size = (self.stackViewer.viewer.size().width(), self.stackViewer.viewer.size().height())
        # print('viewer size ', viewer_size)
        # get the image size
        image_size = (self.width_input_file, self.height_input_file)
        # print('image size ', image_size)

        x = event.pos().x()
        y = event.pos().y()

        # 3 possible scenarios; assume that the input is SQUARE
        if viewer_size[0] > viewer_size[1]: # the width of the viewer is bigger than the height
            # print('Width > height')
            scaling_factor = image_size[1]/viewer_size[1]
            x_offset = (viewer_size[0] - viewer_size[1])/2
            y_coordinate = y*scaling_factor
            x_coordinate = (x-x_offset)*scaling_factor

        elif viewer_size[0] < viewer_size[1]:# the width of the viewer is smaller than the height
            # print('height > width')
            scaling_factor = image_size[0] / viewer_size[0]
            y_offset = (viewer_size[1] - viewer_size[0]) / 2
            y_coordinate = (y-y_offset) * scaling_factor
            x_coordinate = x * scaling_factor

        else:  # the width of the viewer  =  height
            # print('Width = height')
            scaling_factor = image_size[0] / viewer_size[0]
            y_coordinate = y * scaling_factor
            x_coordinate = x * scaling_factor

        # print(f'x/y input,  {x}  {y} new coord:  {x_coordinate}  {y_coordinate}')
        self.w = PixelTemporalVariation(round(x_coordinate), round(y_coordinate), self.filename, self.get_acquisition_rate())
        self.w.show()



if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.argv(app.exec())
