import sys

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    try:
        from PyQt4.QtGui import QApplication
    except ImportError:
        raise ImportError("Requires PyQt5 or PyQt4.")

from MultiPageTIFFViewerQt import MultiPageTIFFViewerQt

if __name__ == '__main__':
    # Create the QApplication.
    app = QApplication(sys.argv)

    # Create an image stack viewer widget.
    stackViewer = MultiPageTIFFViewerQt()

    # MultiPageTIFFViewerQt.py.viewer references
    # the stack viewer's ImageViewerQt child widget.
    # Thus, in this example we can set the
    # display and mouse interaction options
    # as described for ImageViewerQt via the
    # stackViewer.viewer reference.
    # Or just accept the defaults as we do here.

    # Load an image stack to be displayed.
    # With no arguments, loadImageStack() will popup
    # a file selection dialog. Optionally, you can
    # call loadImageStack(fileName) directly as well.
    stackViewer.loadImageStackFromFile('im1.tif')

    # Read the entire stack into memory.
    # For large stacks this can be time and memory hungry.
    # !!! ONLY do this if you REALLY NEED TO !!!
    # For example, if you need to do a z projection
    # over the stack. Otherwise, this is a waste
    # of time and memory, and it's NOT necessary
    # just to view the stack as subsequent frames
    # will be loaded as they're needed in the viewer.
    # !!! Currently, this ONLY works for grayscale image
    # stacks that can be represented as 3D arrays.
    # Uncomment the following line if you want this.
    # entireStackArray = stackViewer.getAllFrames()

    # Show the viewer and run the application.
    stackViewer.show()
    sys.exit(app.exec_())