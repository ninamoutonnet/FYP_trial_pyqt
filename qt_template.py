import sys
from PyQt5 import QtWidgets as qtw
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import tiffcapture
import qimage2ndarray


class MainWindow(qtw.QWidget):

    def __init__(self):
        '''Main Window constructor'''
        super().__init__('Calcium Analysis GUI')
        # main UI code goes here

        # create the window
        self.setWindowTitle("PyQt5 Media Player")
        self.setGeometry(350, 100, 700, 500)
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)
        self.init_ui()

        # end main UI code
        self.show()


if __name__=='__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.argv(app.exec())


