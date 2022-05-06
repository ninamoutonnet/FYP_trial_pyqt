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


class PixelTemporalVariation(qtw.QWidget):

    def __init__(self, x, y, filename):
        super().__init__()
        #  start UI code
        # QLabel
        label = qtw.QLabel('Pixel Variation 1D', self, margin=10)

        # give the tiff file to the intensity function
        # print('pixel x: ', x , 'and y: ', y, 'of file: ', filename)

        # Read the image from the TIFF file as numpy array:
        imfile = tifffile.imread(filename)
        # print('shape: ', imfile.shape)
        extracted = imfile[:, x, y]
        # print('shape: ', extracted.shape)

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

    def __init__(self, filename):
        '''Main Window constructor'''
        super().__init__()
        #  get the filname/path of the original tiff stack to use when looking at individual pixel intensity
        self.filename = filename

        #  start UI code
        # QLabel
        label = qtw.QLabel(filename, self, margin=10)

        # give the tiff file to the intensity function
        self.fluo_output = fluoMap(filename)
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
        downsample_value_widget = qtw.QLineEdit()
        #downsample_value_widget.setValidator(qtw.QIntValidator())  # check that the downsample is an integer
        downsample_value_widget.setMaxLength(self.width_input_file) # check that the downsample is not above the current size of the picture

        sampling_rate_value_widget = qtw.QLineEdit()
        #sampling_rate_value_widget.setValidator(qtw.QIntValidator())  # check that the sampling rate is an integer

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
        variable_layout.addRow('Downsampling value', downsample_value_widget)
        variable_layout.addRow('Acquisition rate in Hz', sampling_rate_value_widget)
        right_layout.addWidget(metadata_widget)

        simple_processing_widget = qtw.QGroupBox('Quick overview')
        right_layout.addWidget(simple_processing_widget)
        simple_processing_layout = qtw.QHBoxLayout()
        simple_processing_widget.setLayout(simple_processing_layout)
        simple_processing_layout.addWidget(button_pixel_intensity_var)
        simple_processing_layout.addWidget(button)

        complex_processing_widget = qtw.QGroupBox('Cell segmentation')
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

    def GetPos(self, event):
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

    def WindowIntensityVariation(self):
        # Add event when the buttons are pressed -> get the position of interest
        self.stackViewer.viewer.mousePressEvent = self.GetPos

    def windowFluo(self):
        self.w = FluorescenceIntensityMap(self.filename)


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.argv(app.exec())
