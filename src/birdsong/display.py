import sys
#from PyQt5.QtWidgets import *
#from PyQt5.QtGui import *

from PyQt5.QtWidgets import QMainWindow, QApplication 
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout

                            

import matplotlib
matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvas

#*****from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
#******import seaborn as sns
#import math
# from sklearn.metrics import average_precision_score

# LOG_FILEPATH = '/Users/LeoGl/Documents/bird/15-10-2020_17-37_K7_B32.jsonl'
# LOG_FILEPATH = '/Users/LeoGl/Documents/bird/29-10-2020_12-59_K7_B32.jsonl'
# LOG_FILEPATH = '/Users/LeoGl/Documents/bird/fullAugmentedSongAndCall_K7_B32.jsonl'
# LOG_FILEPATH = '/Users/LeoGl/Documents/bird/logs/05-09-2020_18-17_K7_B128.jsonl'
#LOG_FILEPATH = '/Users/paepcke/EclipseWorkspacesNew/birds/src/birdsong/runs_json/12-01-2021_21-30_K7_B32.jsonl'
#LOG_FILEPATH = '/Users/paepcke/EclipseWorkspacesNew/birds/tmp/11-12-2020_13-57_K7_B64.jsonl'
LOG_FILEPATH = '/Users/paepcke/EclipseWorkspacesNew/birds/tmp/30-10-2020_13-05_K7_B32WithFiles.jsonl'

WIDTH = 3000
HEIGHT = 2000

class App(QMainWindow):
    
    #------------------------------------
    # App Constructor 
    #-------------------
    
    def __init__(self, LOG_FILEPATH):
        super().__init__()
        # initialize the file reader
        self.data = LogReader(LOG_FILEPATH)
        
        #*****
        # self.cross_check(LOG_FILEPATH, self.data)
        #*****

        self.title = 'Birdsong Classifier'
        self.left = 0
        self.top = 0
        self.width = WIDTH
        self.height = HEIGHT

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.allwidget = OverallWidget(self, self.data)

        self.setCentralWidget(self.allwidget)
        self.show()

    #------------------------------------
    # cross_check 
    #-------------------

    def cross_check(self, filepath, data):
        
        with open(filepath, 'r') as f:
            
            all_lines = f.readlines()
            assert(len(all_lines) == data.confusion.sum())

# ----------------------------- Class LogReader ---------        

class LogReader():
    
    #------------------------------------
    # Constructor LogReader
    #-------------------
    
    def __init__(self, LOG_FILEPATH):
        self.LOG_FILEPATH = LOG_FILEPATH

        self.splitList = None

        self.epoch = []
        self.loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.confusion = None
        self.normal = None
        self.kernel = None
        self.batch = None
        self.MAP = None
        self.files = [[[] for b in range(20)] for m in range(20)]
        self.getKernelBatch()

        self.readLine()

    #------------------------------------
    # getKernelBatch 
    #-------------------

    def getKernelBatch(self):
        kernel_index = self.LOG_FILEPATH.find('K')
        end_index = self.LOG_FILEPATH.find('.')
        kernelbatch = self.LOG_FILEPATH[kernel_index: end_index].split('_')
        self.kernel = kernelbatch[0][1:]
        self.batch = kernelbatch[1][1:]

    #------------------------------------
    # readLine 
    #-------------------

    def readLine(self):

        with open(self.LOG_FILEPATH, 'r') as f:
            
            # Read header:
            col_header_str = f.readline()
            # Get an array of column names:
            col_header = self.safe_eval(col_header_str)

            for line in f:
                # Get an array of column values
                # for the current row:
                log_cols  = self.safe_eval(line)
                
                # A dictionary comprehension: line
                # up col header names with col values
                # (the zip() expression), and make a
                # dict {col_name : col_value}:
                
                log_entry = {col_name : val for col_name,val in zip(col_header, log_cols)}
                
                self.epoch.append(log_entry['epoch'])
                self.loss.append(log_entry['loss'])
                self.train_accuracy.append(log_entry['training_accuracy'])
                self.test_accuracy.append(log_entry['testing_accuracy'])

                self.confusion =  np.array(log_entry['confusion_matrix'])
                    
# ************* Remove when clear about meaning
#                 if len(self.splitlist) > 2:
#                     two_dim_list = self.splitlist[2:21]
#                     x = 0
#                     for row in two_dim_list:
#                         y = 0
#                         for cell in row.split('], ['):
#                             self.files[x][y].append(cell)
#                             y += 1
#                         x += 1
# *************

        self.genTru()
        self.calcNorm()
        self.calcMAP()

    #------------------------------------
    # genTru
    #-------------------

    def genTru(self):
        self.tru = self.confusion.copy()
        for pred in range(len(self.confusion)):
            for truth in range(len(self.confusion[0])):
                if truth != pred:
                    self.tru[truth][truth] += self.confusion[pred, truth]
        for pred in range(len(self.confusion)):
            for truth in range(len(self.confusion[0])):
                if truth != pred:
                    self.tru[pred, truth] = 0

    #------------------------------------
    # calcNorm 
    #-------------------

    def calcNorm(self):
        self.normal = self.confusion.copy()
        self.normal[0:len(self.normal)-1][0:len(self.normal[0])-1] = 0
        for pred in range(len(self.confusion)):
            for truth in range(len(self.confusion[0])):
                if self.tru[pred][pred] != 0:
                    self.normal[pred][truth] = self.confusion[pred][truth] / self.tru[pred][pred]

    #------------------------------------
    # calcMAP 
    #-------------------

    def calcMAP(self):
        temp = self.confusion.copy()
        for pred in range(len(self.confusion)):
            for truth in range(len(self.confusion[0])):
                if pred == truth:
                    temp[pred][truth] = 0
        false_pos = temp.sum(axis=0)
        false_positives = np.zeros(int(len(false_pos) / 2))
        count = 0
        for species in range(len(false_pos)):
            if species % 2 == 0:
                false_positives[count] = false_pos[species]
                count += 1
        count = 0
        for species in range(len(false_pos)):
            if species % 2 != 0:
                false_positives[count] += false_pos[species]
                count += 1
        temp = self.confusion.copy()
        for pred in range(len(self.confusion)):
            for truth in range(len(self.confusion[0])):
                if pred != truth:
                    temp[pred][truth] = 0
        true_pos = temp.sum(axis=0)
        true_positives = np.zeros(int(len(true_pos) / 2))
        count = 0
        for species in range(len(true_pos)):
            if species % 2 == 0:
                true_positives[count] = true_pos[species]
                count += 1
        count = 0
        for species in range(len(true_pos)):
            if species % 2 != 0:
                true_positives[count] += true_pos[species]
                count += 1

        denom = np.add(true_positives, false_positives)
        y_scores = np.true_divide(true_positives, denom)

        print("average precision scores by class", y_scores)
        self.MAP = np.sum(y_scores) / len(y_scores)
        print("mean average precision", self.MAP)

    #------------------------------------
    # safe_eval
    #-------------------

    def safe_eval(self, item):
        '''
        Given a string that can be interpreted
        as a Python expression, evaluate the expression
        safely, and return the result.
        
        Evaluating safely means that no Python 
        is available to the eval process: not 
        built-in, or library functions.  
        
        @param item: a Python expression
        @type item: str
        @return: evaluated Python
        @rtype: Any
        '''
        res = eval(item,
               {"__builtins__":None},    # No built-ins at all
               {}                        # No additional func
               )
        return res
        

    # --------------------- Getters ----------------
    def getMAP(self):
        return self.MAP

    def getEpoch(self):
        return self.epoch

    def getLoss(self):
        return self.loss

    def getTrainingAccuracy(self):
        return self.train_accuracy

    def getTestingAccuracy(self):
        return self.test_accuracy

    def getConfusion(self):
        return self.confusion

    def getKernel(self):
        return self.kernel

    def getBatch(self):
        return self.batch

    def getNormalized(self):
        return self.normal

    def getFileMatrix(self):
        return self.files


# ----------------------------- Class OverallWidget ---------

class OverallWidget(QWidget):
    def __init__(self, parent, file):

        super(QWidget, self).__init__(parent)
        self.layout = QHBoxLayout(self)

        self.leftwidget = LeftSideWidget(self, file)
        self.rightwidget = RightSideWidget(self, file)

        self.layout.addWidget(self.leftwidget)
        self.layout.addWidget(self.rightwidget)

# ----------------------------- Class LeftSideWidget ---------

class LeftSideWidget(QWidget):
    def __init__(self, parent, file):

        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.losswidget = PlotLoss(self, file)
        self.accuracywidget = PlotAccuracy(self, file)

        self.layout.addWidget(self.losswidget)
        self.layout.addWidget(self.accuracywidget)

# ----------------------------- Class PlotLoss ---------

class PlotLoss(QWidget):
    def __init__(self, parent, file):

        super(QWidget, self).__init__(parent)
        self.figure = plt.figure(figsize=(10,5))
        self.resize(300,300)

        self.canvas = FigureCanvas(self.figure)
        ax = self.figure.add_subplot(111)
        ax.set_title('Loss vs Epoch')
        ax.set_ylim(0, 10)
        ax.plot(file.getEpoch(), file.getLoss(), 'r')
        self.canvas.draw()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        

# ----------------------------- Class PlotAccuracy ---------

class PlotAccuracy(QWidget):
    def __init__(self, parent, file):

        super(QWidget, self).__init__(parent)
        self.figure = plt.figure(figsize=(10,5))
        self.resize(300,300)

        self.canvas = FigureCanvas(self.figure)
        ax = self.figure.add_subplot(111)
        ax.set_title('Testing & Training Accuracy vs Epoch')
        ax.set_ylim(0, 100)
        ax.plot(file.getEpoch(), file.getTrainingAccuracy(), 'b')
        ax.plot(file.getEpoch(), file.getTestingAccuracy(), 'r')
        self.canvas.draw()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

# ----------------------------- Class RightSideWidget ---------

class RightSideWidget(QWidget):
    def __init__(self, parent, file):

        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.file_label = QLabel()
        self.file_list = QLineEdit()
        self.textwidget = QLabel()

        self.textwidget.setFont(QFont('Arial', 20)) 
        self.textwidget.setFixedWidth(WIDTH * 0.4)

        self.file = file
        self.confusionwidget = PlotConfusion(self, file)
        self.confusionwidget.setFixedWidth(WIDTH * 0.4)
        self.confusionwidget.setFixedHeight(HEIGHT * 0.7)
        self.textwidget.setText('  Kernel Size is ' + file.getKernel() + '\n  Batch size is ' + file.getBatch()
                                + '\n  MAP is ' + str(self.file.getMAP())[:7])

        self.layout.addWidget(self.textwidget)
        self.layout.addWidget(self.confusionwidget)
        self.cid = self.confusionwidget.canvas.mpl_connect("button_release_event", self.onRelease)

    #------------------------------------
    # onRelease 
    #-------------------

    def onRelease(self, event):
        actual = self.confusionwidget.axis_labels[self.confusionwidget.xint]
        predicted = self.confusionwidget.axis_labels[self.confusionwidget.yint]
        norm_acc = self.confusionwidget.file.getNormalized()[self.confusionwidget.yint][self.confusionwidget.xint]
        file_names = self.confusionwidget.file.getFileMatrix()[self.confusionwidget.yint][self.confusionwidget.xint]
        if file_names != []:
            self.file_label.setText('file names of misclassified samples: ')
            self.file_list.setText(str(file_names[0])[1:-1])
            self.file_list.setFont(QFont('Arial', 10))
            self.file_list.setFixedWidth(WIDTH * 0.4)
            self.file_label.setFont(QFont('Arial', 10))
            self.file_label.setFixedHeight(50)
            self.file_label.setFixedWidth(WIDTH * 0.4)
            self.layout.addWidget(self.file_label)
            self.layout.addWidget(self.file_list)
        else:
            self.file_list.setText('')

        self.textwidget.setText('  Actual Species:  ' + str(actual) + "               Kernel Size:" + str(self.file.getKernel()) + '\n  Predicted Species:  ' + str(predicted) + "               Batch Size:" + str(self.file.getBatch()) + '\n  Normalized Accuracy:  ' + str(norm_acc))
        self.textwidget.setFont(QFont('Arial', 10))


# ----------------------------- Class PlotConfusion ---------

class PlotConfusion(QWidget):
    
    #------------------------------------
    # Constructor PlotConfusion 
    #-------------------
    
    def __init__(self, parent, file):

        # axis_labels = ['AMADEC', 'ARRAUR','CORALT','DYSMEN', 'EUPIMI','HENLES','HYLDEC','LOPPIT', 'TANGYR', 'TANICT']
        self.axis_labels = ['AMADEC_CALL', 'AMADEC_SONG', 'ARRAUR_CALL', 'ARRAUR_SONG', 'CORALT_CALL', 'CORALT_SONG', 'DYSMEN_CALL', 'DYSMEN_SONG', 'EUPIMI_CALL',
                        'EUPIMI_SONG', 'HENLES_CALL', 'HENLES_SONG', 'HYLDEC_CALL', 'HYLDEC_SONG', 'LOPPIT_CALL', 'LOPPIT_SONG', 'TANGYR_CALL', 'TANGYR_SONG',
                        'TANICT_CALL', 'TANICT_SONG']

        super(QWidget, self).__init__(parent)
        self.figure = plt.figure(figsize=(10, 5))
        self.resize(400, 400)

        self.canvas = FigureCanvas(self.figure)
        self.xint = -1
        self.yint = -1
        self.file = file

        ax = self.figure.add_subplot(111)
        ax.set_title('Confusion Matrix on Last Epoch')
        # ax = sns.heatmap(file.getConfusion(), xticklabels=self.axis_labels, yticklabels=axis_labels, center=10, vmax=20)
        ax = sns.heatmap(file.getNormalized(), xticklabels=self.axis_labels, yticklabels=self.axis_labels, center = 0.45)
        ax.set_xlabel('actual species')
        ax.set_ylabel('predicted species')

        ax.tick_params(axis='x', labelrotation=90)

        self.canvas.draw()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.cid = self.canvas.mpl_connect("motion_notify_event", self.onMotion)
        self.heatmap = ax

    #------------------------------------
    # onMotion 
    #-------------------

    def onMotion(self, event):
        if not event.inaxes:
            self.xint = -1
            self.yint = -1
            return

        self.xint = int(event.xdata)
        self.yint = int(event.ydata)
        self.rect = mpatches.Rectangle((self.xint, self.yint), 1, 1, fill=False, linestyle='dashed', edgecolor='red',
                                       linewidth=2.0)

        self.heatmap.add_patch(self.rect)
        self.canvas.draw()
        self.rect.remove()

# ---------------------- Main --------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    x = App(LOG_FILEPATH)
    sys.exit(app.exec_())

