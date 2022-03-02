import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from MultiPageTIFFViewerQt import MultiPageTIFFViewerQt


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

        # Add widget objects to a layout
        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(stackViewer)


        # end main UI code - Display the UI
        self.show()


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.argv(app.exec())





