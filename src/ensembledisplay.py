import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import random
import csv
import pandas as pd
import seaborn as sns
import json
import math


#For the final accuracy to line up, this MUST be the newest ensemble output
LOG_FILEPATH = 'ensembleoutput.jsonl'
NEWEST_ENSEMBLE_FILEPATH = 'newestensembleoutput.txt'

class App(QMainWindow):
	def __init__(self, LOG_FILEPATH):
		super().__init__()
		#initialize the file reader
		self.file = FileRead(LOG_FILEPATH)


		self.title = 'Ensemble Birdsong Classifier'
		self.left = 0
		self.top = 0
		self.width = 1500
		self.height = 600

		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.allwidget = OverallWidget(self, self.file)

		self.setCentralWidget(self.allwidget)
		self.show()


class FileRead():
	def __init__(self, LOG_FILEPATH):
		self.LOG_FILEPATH = LOG_FILEPATH
		self.NEWEST_ENSEMBLE_FILEPATH = NEWEST_ENSEMBLE_FILEPATH

		self.splitList = None
		self.new_splitList = None

		
		self.epoch = [[], [], [], []]
		self.netnumber = [[], [], [], []]
		self.loss = [[], [], [], []]
		self.train_accuracy = [[], [], [], []]
		self.confusion = []

		with open(self.LOG_FILEPATH, 'r') as f:
			f.readline() #read header line of json file
		self.readLine()

	def readLine(self):
		with open(self.LOG_FILEPATH, 'r') as f:
			f.readline()

			temp_netnumber = 1
			not_EOF = True

			while not_EOF:
				line = f.readline()
				netnumber_change = False

				#set EOF flag; do not go for another round
				if line == '':
					print("reached EOF")
					not_EOF = False

				#does the confusion matrix need to be stored? 
				if line != '':
					self.new_splitlist = line.split('[[')
					split2 = self.new_splitlist[0].split('[')
					data = split2[1].split(',')
					data[1] = data[1][-3]
					if (temp_netnumber != int(data[1])):
						netnumber_change = True

					temp_netnumber = int(data[1])

				#store the confusion matrix
				if line == '' or netnumber_change:
					con = self.splitlist[1][:-4] #second half of previous round's splitlist, ie the confusion matrix
					consplit = con.split('], [')
					confusion = np.fromstring(consplit[0], sep = ',')
					for i in range (1, len(consplit)):
						confusion = np.concatenate((confusion, np.fromstring(consplit[i], sep = ',')))
					dim = int(math.sqrt(confusion.shape[0]))
					confusion = confusion.reshape((dim, dim))
					print("append" + str(confusion))
					self.confusion.append(confusion)

				#store all other variables
				else:
					#data[1]-1 term accounts for netnumber
					self.splitlist = self.new_splitlist

					self.epoch[temp_netnumber - 1].append(data[0])
					self.netnumber.append(data[1])
					self.loss[temp_netnumber - 1].append(data[2])
					self.train_accuracy[temp_netnumber - 1].append(data[3])

			print(self.train_accuracy)

	def getEpoch(self, netnumber):
		return self.epoch[netnumber - 1]

	def getLoss(self, netnumber):
		return self.loss[netnumber - 1]

	def getTrainingAccuracy(self, netnumber):
		return self.train_accuracy[netnumber - 1]

	def getConfusion(self, netnumber):
		return self.confusion [netnumber - 1]


class OverallWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QHBoxLayout(self)

		self.leftwidget = LeftSideWidget(self, file)
		self.middlewidget = MiddleWidget(self, file)
		self.rightwidget = RightSideWidget(self, file)

		self.layout.addWidget(self.leftwidget)
		self.layout.addWidget(self.middlewidget)
		self.layout.addWidget(self.rightwidget)

class LeftSideWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

		self.accuracywidget1 = PlotAccuracy(self, file, 1)
		self.accuracywidget3 = PlotAccuracy(self, file, 3)

		self.layout.addWidget(self.accuracywidget1)
		self.layout.addWidget(self.accuracywidget3)

class MiddleWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

		self.accuracywidget2 = PlotAccuracy(self, file, 2)
		self.accuracywidget4 = PlotAccuracy(self, file, 4)

		self.layout.addWidget(self.accuracywidget2)
		self.layout.addWidget(self.accuracywidget4)


class PlotAccuracy(QWidget):
	def __init__(self, parent, file, netnumber):

		super(QWidget, self).__init__(parent)
		self.figure = plt.figure(figsize=(10,5))
		self.resize(300,300)

		self.canvas = FigureCanvas(self.figure)
		ax = self.figure.add_subplot(111)
		ax.set_title('Training Accuracy vs Epoch: Net ' + str(netnumber))
		ax.set_ylim(0, 100)

		ax.plot(file.getEpoch(netnumber), np.array(file.getTrainingAccuracy(netnumber), dtype=np.float32), 'b')
		#plt.xticks(np.arange(0, 60, 5))
		#plt.yticks(np.arange(0, 100, 10))
		print(file.getTrainingAccuracy(netnumber))
		loc = plticker.MultipleLocator(base=10.0) # this locator puts ticks at regular intervals
		ax.xaxis.set_major_locator(loc)
		ax.yaxis.set_major_locator(loc)

		self.canvas.draw()

		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		self.setLayout(layout)


class RightSideWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

		with open(file.NEWEST_ENSEMBLE_FILEPATH, 'r') as f:
		    lines = f.read().splitlines()
		    last_line = lines[-1]
		    final_accuracy = last_line.split('accuracy is')[1]

		self.textwidget = QLabel()
		self.textwidget.setText("Final testing accuracy of newest ensemble output is: " + 
			str(round(float(final_accuracy), 3)) + "%")
		self.textwidget.setFont(QFont('Arial', 20)) 
		self.textwidget.resize(200, 150)

		self.confusionwidget = TabConfusion(self, file)

		self.layout.addWidget(self.textwidget)
		self.layout.addWidget(self.confusionwidget)


class TabConfusion(QWidget):
	#written in a hurry, apologies
	def __init__(self, parent, file):
		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)
		
		# Initialize tab screen
		self.tabs = QTabWidget()
		self.tabs.resize(200,300)

		# Add tabs
		self.tab1 = QWidget()
		self.tab2 = QWidget()
		self.tab3 = QWidget()
		self.tab4 = QWidget()

		self.tabs.addTab(self.tab1, "Net 1")
		self.tabs.addTab(self.tab2, "Net 2")
		self.tabs.addTab(self.tab3, "Net 3")
		self.tabs.addTab(self.tab4, "Net 4")

		#configure tab layout
		self.tab1.layout = QVBoxLayout(self)
		self.confusion1 = PlotConfusion(self, file, 1)
		self.tab1.layout.addWidget(self.confusion1)
		self.tab1.setLayout(self.tab1.layout)

		self.tab2.layout = QVBoxLayout(self)
		self.confusion2 = PlotConfusion(self, file, 2)
		self.tab2.layout.addWidget(self.confusion2)
		self.tab2.setLayout(self.tab2.layout)

		self.tab3.layout = QVBoxLayout(self)
		self.confusion3 = PlotConfusion(self, file, 3)
		self.tab3.layout.addWidget(self.confusion3)
		self.tab3.setLayout(self.tab3.layout)

		self.tab4.layout = QVBoxLayout(self)
		self.confusion4 = PlotConfusion(self, file, 4)
		self.tab4.layout.addWidget(self.confusion4)
		self.tab4.setLayout(self.tab4.layout)

		# Add tabs to widget
		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)


class PlotConfusion(QWidget):
	def __init__(self, parent, file, netnumber):

		axis_labels = [['DYSMEN_S', 'HENLES_C','HENLES_S','LOPPIT'], 
			['ARRAUR_C','ARRAUR_S','DYSMEN_C','HYLDEC'], 
			['CORALT_C', 'CORALT_S', 'TANICT'],
			['AMADEC', 'EUPIMI', 'TANGYR']]

		super(QWidget, self).__init__(parent)
		self.figure = plt.figure(figsize=(10,5))
		self.resize(400,400)

		self.canvas = FigureCanvas(self.figure)

		ax = self.figure.add_subplot(111)
		ax.set_title('Confusion Matrix on Last Epoch')
		ax = sns.heatmap(file.getConfusion(netnumber), xticklabels = axis_labels[netnumber - 1], 
			yticklabels = axis_labels[netnumber - 1])

		ax.tick_params(axis = 'x', labelrotation = 45, labelsize = 8)
		ax.tick_params(axis = 'y', labelrotation = 45, labelsize = 8)

		self.canvas.draw()

		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		self.setLayout(layout)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	x = App(LOG_FILEPATH)
	sys.exit(app.exec_())

