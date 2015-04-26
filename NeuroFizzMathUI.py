#!/usr/bin/env python
# User interface for NeuroFizzMath program

import NeuroFizzMath
import sys
from PyQt4 import QtGui


class Cheese(QtGui.QMainWindow):
    
    def __init__(self):
        super(Cheese, self).__init__()
        
        self.initUI()
        
        
    def initUI(self):               
        
        textEdit = QtGui.QTextEdit()
        self.setCentralWidget(textEdit)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        saveAction = QtGui.QAction(QtGui.QIcon('save.png'), 'Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save')
        saveAction.triggered.connect(self.saveState)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction('New')
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Models')
        fileMenu.addAction('Fitzhugh-Nagumo')
        fileMenu.addAction('Morris-Lecar')
        fileMenu.addAction('Izikevich')
        fileMenu.addAction('Hindmarsh-Rose')
        fileMenu.addAction('Hodgkins-Huxley')

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Plots')
        fileMenu.addAction('Phase Plot')
        fileMenu.addAction('Membrane Potential over Time')
        fileMenu.addAction('FFT')

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)
        toolbar = self.addToolBar('Save')
        toolbar.addAction(saveAction)
        
        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('NeuroFizzMath')
        self.show()
        
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Cheese()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    