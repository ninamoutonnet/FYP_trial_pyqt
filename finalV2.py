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
import CNMFE


class PixelTemporalVariation(qtw.QWidget):

    def __init__(self, x, y, filename):
        super().__init__()
        #  start UI code
        # QLabel
        label = qtw.QLabel('Pixel Variation 1D', self, margin=10)

        # Read the image from the TIFF file as numpy array:
        imfile = tifffile.imread(filename)
        extracted = imfile[:, x, y]

        graphWidget = pg.PlotWidget()
        pen = pg.mkPen(color=(255, 0, 0), width=5)
        graphWidget.setBackground('w')
        graphWidget.plot(extracted, pen=pen)
        graphWidget.showGrid(x=True, y=True)
        graphWidget.setTitle("Temporal Trace of ("+str(x)+","+str(y)+")", size="30pt")
        styles = {'color': 'b', 'font-size': '20px'}
        graphWidget.setLabel('left', 'Intensity normalised by the average intensity', **styles)
        graphWidget.setLabel('bottom', 'Frame number', **styles)

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(label)
        layout.addWidget(graphWidget)

        #  end main UI code - Display the UI
        self.show()


class FluorescenceIntensityMap(qtw.QWidget):

    def __init__(self, filename, downsample_factor):

        '''Main Window constructor'''
        super().__init__()
        #  get the filname/path of the original tiff stack to use when looking at individual pixel intensity
        self.filename = filename

        #  start UI code
        # QLabel
        label = qtw.QLabel(filename, self, margin=10)

        # give the tiff file to the intensity function
        self.fluo_output = fluoMap(filename, downsample_factor)
        self.stackViewer = MultiPageTIFFViewerQt()
        # As this next command has no argument, the files will pop up and the
        # user will be asked to get the tiff stack to open
        self.stackViewer.loadImageStackFromFile(self.fluo_output)

        # Create a button to see 1D intensity variation
        # QPushButton
        button = qtw.QPushButton(
            "Normalised pixel temporal intensity variation",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.stackViewer)
        layout.addWidget(label)
        layout.addWidget(button)

        #  end main UI code - Display the UI
        self.show()

        #  If the button is pressed, open a new window
        button.clicked.connect(self.WindowIntensityVariation)

    def GetPos(self, event):
        # get the x pixel correction
        x = event.pos().x()
        y = event.pos().y()
        print('x= ', x)
        print('y= ', y)
        self.w = PixelTemporalVariation(x, y, self.fluo_output) # self.filename)
        self.w.show()

    def WindowIntensityVariation(self):
        # Add event when the buttons are pressed -> get the position of interest
        self.stackViewer.viewer.mousePressEvent = self.GetPos


class CNMFE_GUI(qtw.QWidget):

    def __init__(self, filename, downsample_factor):

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

        # create the text widgets for the entry of the downsampling factor and sampling rate
        self.average_cell_diameter = qtw.QLineEdit()
        label_average_cell_diameter = qtw.QLabel('Average cell diameter')
        label_average_cell_diameter.setToolTip('The average cell diameter of a representative cell in pixels,  <b> default = 7 <b>')
        variable_layout.addRow(label_average_cell_diameter, self.average_cell_diameter)

        self.min_pixel_correlation = qtw.QLineEdit()
        label_min_pixel_correlation = qtw.QLabel('Minimum pixel correlation')
        label_min_pixel_correlation.setToolTip('The minimum correlation of a pixel with its immediate neighbors when searching for new cell centers,  <b> default = 0.8 <b>')
        variable_layout.addRow(label_min_pixel_correlation, self.min_pixel_correlation)

        self.min_peak_to_noise_ratio = qtw.QLineEdit()
        label_min_peak_to_noise_ratio = qtw.QLabel('min_peak_to_noise_ratio')
        label_min_peak_to_noise_ratio.setToolTip('The minimum peak-to-noise ratio of a pixel when searching for new cell centers,  <b> default = 10 <b>')
        variable_layout.addRow(label_min_peak_to_noise_ratio, self.min_peak_to_noise_ratio)

        self.gaussian_kernel_size = qtw.QLineEdit()
        label_gaussian_kernel_size = qtw.QLabel('Gaussian kernel size')
        label_gaussian_kernel_size.setToolTip('The width in pixels of the Gaussian kernel used for spatial filtering of the movie before cell initialization (automatically estimated when the value provided is smaller than 3),  <b> default = 0 <b>')
        variable_layout.addRow(label_gaussian_kernel_size, self.gaussian_kernel_size)

        self.closing_kernel_size = qtw.QLineEdit()
        label_closing_kernel_size = qtw.QLabel('Closing kernel size')
        label_closing_kernel_size.setToolTip('The size in pixels of the morphological closing kernel used for removing small disconnected components and connecting small cracks within individual cell footprints (automatically estimated when the value provided is smaller than 3),  <b> default = 0 <b>')
        variable_layout.addRow(label_closing_kernel_size, self.closing_kernel_size)

        self.background_downsampling_factor = qtw.QLineEdit()
        label_background_downsampling_factor = qtw.QLabel('Background downsampling factor')
        label_background_downsampling_factor.setToolTip('The spatial downsampling factor to use when estimating the background activity,  <b> default = 2 <b>')
        variable_layout.addRow(label_background_downsampling_factor, self.background_downsampling_factor)

        self.ring_size_factor = qtw.QLineEdit()
        label_ring_size_factor = qtw.QLabel('Ring size factor')
        label_ring_size_factor.setToolTip('The multiple of the average cell diameter to use for computing the radius of the ring model used for estimating the background activity,  <b> default = 1.4 <b>')
        variable_layout.addRow(label_ring_size_factor, self.ring_size_factor)

        self.merge_threshold = qtw.QLineEdit()
        label_merge_threshold = qtw.QLabel('Merge threshold')
        label_merge_threshold.setToolTip('The temporal correlation threshold for merging cells that are spatially close,  <b> default = 0.7 <b>')
        variable_layout.addRow(label_merge_threshold, self.merge_threshold)

        self.num_threads = qtw.QLineEdit()
        label_num_threads = qtw.QLabel('Number of threads')
        label_num_threads.setToolTip('The number of threads to use for processing	,  <b> default = 4 <b>')
        variable_layout.addRow(label_num_threads, self.num_threads)

        self.processing_mode = qtw.QLineEdit()
        label_processing_mode = qtw.QLabel('Processing mode')
        label_processing_mode.setToolTip('the processing mode to use to run CNMF-E (0: all in memory, 1: sequential patches, 2: parallel patches). All in memory: processes the entire field of view at once. Sequential patches: breaks the field of view into overlapping patches and processes them one at a time using the specified number of threads where parallelization is possible. Parallel patches: breaks the field of view into overlapping patches and processes them in parallel using a single thread for each.,  <b> default = 2 <b>')
        variable_layout.addRow(label_processing_mode, self.processing_mode)

        self.patch_size = qtw.QLineEdit()
        label_patch_size = qtw.QLabel('Patch size')
        label_patch_size.setToolTip('The side length of an individual square patch of the field of view in pixels,  <b> default = 80 <b>')
        variable_layout.addRow(label_patch_size, self.patch_size)

        self.patch_overlap = qtw.QLineEdit()
        label_patch_overlap = qtw.QLabel('Patch overlap')
        label_patch_overlap.setToolTip('The amount of overlap between adjacent patches in pixels,  <b> default = 20 <b>')
        variable_layout.addRow(label_patch_overlap, self.patch_overlap)

        self.deconvolve = qtw.QLineEdit()
        label_deconvolve = qtw.QLabel('Deconvolve')
        label_deconvolve.setToolTip('Specifies whether to deconvolve the final temporal traces (0: return raw traces, 1: return deconvolved traces),  <b> default = 0 <b>')
        variable_layout.addRow(label_deconvolve, self.deconvolve)

        self.output_units = qtw.QLineEdit()
        label_output_units = qtw.QLabel('Output units')
        label_output_units.setToolTip('The units of the output temporal traces (0: dF, 1: dF over noise). dF: temporal traces on the same scale of pixel intensity as the original movie. dF is calculated as the average fluorescence activity of all pixels in a cell, scaled so that each spatial footprint has a magnitude of 1. dF over noise: temporal traces divided by their respective estimated noise level. This can be interpreted similarly to a z-score, with the added benefit that the noise is a more robust measure of the variance in a temporal trace compared to the standard deviation.,  <b> default = 1 <b>')
        variable_layout.addRow(label_output_units, self.output_units)

        self.output_filetype = qtw.QLineEdit()
        label_output_filetype = qtw.QLabel('Output filetype')
        label_output_filetype.setToolTip('The file types into which the output will be saved (0: footprints saved to a tiff file and traces saved to a csv file, 1: output saved to a h5 file under the keys footprints and traces),  <b> default = 0 <b>')
        variable_layout.addRow(label_output_filetype, self.output_filetype)

        self.verbose = qtw.QLineEdit()
        label_verbose = qtw.QLabel('Verbose')
        label_verbose.setToolTip('To enable and disable verbose mode. When enabled, progress is displayed in the console. (0: disabled, 1: enabled),  <b> default = 0 <b>')
        variable_layout.addRow(label_verbose, self.verbose)

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
        # button.clicked.connect(self.WindowIntensityVariation)


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

        # Create a button for each cell sorting algorithm
        button_CNMFE = qtw.QPushButton(
            "CNMFE",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        button_SVD = qtw.QPushButton(
            "SVD",
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
        complex_processing_layout.addWidget(button_SVD)
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

        button_SVD.setSizePolicy(
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

    def cnmfe_trial(self):
        self.i = CNMFE_GUI(self.filename, 2)
        return

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
        if downsample is not None:
            self.w = FluorescenceIntensityMap(self.filename, downsample)

    def getPosition(self, event):
        # get the viewer size
        viewer_size = (self.stackViewer.viewer.size().width(), self.stackViewer.viewer.size().height())
        print('viewer size ', viewer_size)
        # get the image size
        image_size = (self.width_input_file, self.height_input_file)
        print('image size ', image_size)

        x = event.pos().x()
        y = event.pos().y()

        # 3 possible scenarios; assume that the input is SQUARE
        if viewer_size[0] > viewer_size[1]: # the width of the viewer is bigger than the height
            print('Width > height')
            scaling_factor = image_size[1]/viewer_size[1]
            x_offset = (viewer_size[0] - viewer_size[1])/2
            y_coordinate = y*scaling_factor
            x_coordinate = (x-x_offset)*scaling_factor

        elif viewer_size[0] < viewer_size[1]:# the width of the viewer is smaller than the height
            print('height > width')
            scaling_factor = image_size[0] / viewer_size[0]
            y_offset = (viewer_size[1] - viewer_size[0]) / 2
            y_coordinate = (y-y_offset) * scaling_factor
            x_coordinate = x * scaling_factor

        else:  # the width of the viewer  =  height
            print('Width = height')
            scaling_factor = image_size[0] / viewer_size[0]
            y_coordinate = y * scaling_factor
            x_coordinate = x * scaling_factor

        print(f'x/y input,  {x}  {y} new coord:  {x_coordinate}  {y_coordinate}')
        self.w = PixelTemporalVariation(round(x_coordinate), round(y_coordinate), self.filename)
        self.w.show()

    def CNMFE_instance(self):
        # first of all, extract the information from the downsampling and acquisition factor
        downsample_factor = self.get_downsampling_value()
        acquisition_rate = self.get_acquisition_rate()

        if downsample_factor is not None:
            # delete any existing version of the file you are about to over write
            try:
                os.remove('multipage_tif_resized.tif')
                print('removed!')
            except:
                print("An exception occurred")

            # reduce the size of the file
            pages = []
            imagehandler = Image.open(self.filename)
            for page in ImageSequence.Iterator(imagehandler):
                new_size = (int(page.size[0] / downsample_factor), int(page.size[1] / downsample_factor))
                page = page.resize(new_size)
                pages.append(page)
            with TiffImagePlugin.AppendingTiffWriter('multipage_tif_resized.tif') as tf:
                for page in pages:
                    page.save(tf)
                    tf.newFrame()

            #  create a new CNMFE object, give it the resized filename as input
            cnmfe_object = CNMFE.CNMFE_class('multipage_tif_resized.tif')
            #  here the function should automatically display the results, this should be 1 condensed QWindow
            #  The QWindow should have a 'save' button, to enable the scientist to save the data.
            cnmfe_object.plot_summary()


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.argv(app.exec())
