from PyQt5 import QtWidgets

app = QtWidgets.QApplication([])

# create the widget, one of its property is the windowTitle which is set to 'Hello Qt'
window = QtWidgets.QWidget(windowTitle = 'Hello Qt')

# make the widget appear
window.show()

# this line begins the QApplication object event loop. the event loop will run forever until the application quits.
# the app object and the window object do not refer to the same thing, they are linked in the background
# it is important to ensure that a QApplication object exists before creating any QWidget objects
app.exec()