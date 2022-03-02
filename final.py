import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from MultiPageTIFFViewerQt import MultiPageTIFFViewerQt
from ImageProcessing import fluoMap
import os

class fluorescenceIntensityMap(qtw.QWidget):

    def __init__(self, filename):
        '''Main Window constructor'''
        super().__init__()
        #  start UI code
        # QLabel
        label = qtw.QLabel(filename, self, margin=10)

        # give the tiff file to the intensity function
        fluo_output = fluoMap(filename)
        stackViewer = MultiPageTIFFViewerQt()
        # As this next command has no argument, the files will pop up and the
        # user will be asked to get the tiff stack to open
        stackViewer.loadImageStackFromFile(fluo_output)


        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(stackViewer)
        layout.addWidget(label)

        #  end main UI code - Display the UI
        self.show()


class MainWindow(qtw.QWidget):

    def __init__(self):
        '''Main Window constructor'''
        super().__init__()

        # main UI code goes here
        # Create an image stack viewer widget.
        stackViewer = MultiPageTIFFViewerQt()
        # As this next command has no argument, the files will pop up and the
        # user will be asked to get the tiff stack to open
        stackViewer.loadImageStackFromFile()
        # the file path is stored in filename
        self.filename = stackViewer.openFileName

        # Create a button to see the varying intensity.
        # QPushButton
        button = qtw.QPushButton(
            "Fluorescence variation Map",
            self,
            checkable=True,
            checked=True,
            shortcut=qtg.QKeySequence('Ctrl+p')
        )

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(stackViewer)
        layout.addWidget(button)

        #  end main UI code - Display the UI
        self.show()

        #  If the button is pressed, open a new window
        button.clicked.connect(self.windowFluo)


    def windowFluo(self):
        self.w = fluorescenceIntensityMap(self.filename)

if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.argv(app.exec())







