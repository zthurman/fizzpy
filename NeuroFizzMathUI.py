#!/usr/bin/env python
# User interface for program

import NeuroFizzMath
import sys
from PyQt4 import QtGui
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class Cheese(QtGui.QMainWindow):
    
    def __init__(self):
        super(Cheese, self).__init__()
        
        self.initUI()
        
        
    def initUI(self):               
        
        textEdit = QtGui.QTextEdit()
        self.setCentralWidget(textEdit)

        exitAction = QtGui.QAction(QtGui.QIcon('exit24.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(exitAction)
        
        self.setGeometry(300, 300, 350, 250)
        self.setWindowTitle('Main window')    
        self.show()
        
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Cheese()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()    