import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import random
import csv
import pandas as pd
import seaborn as sns
import json
import math



LOG_FILEPATH = '02-09-2020_17-10_K3_B1.jsonl'


class App(QMainWindow):
	def __init__(self, LOG_FILEPATH):
		super().__init__()
		#initialize the file reader
		self.file = FileRead(LOG_FILEPATH)


		self.title = 'Birdsong Classifier'
		self.left = 0
		self.top = 0
		self.width = 1000
		self.height = 500

		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.allwidget = OverallWidget(self, self.file)

		self.setCentralWidget(self.allwidget)
		self.show()


class FileRead():
	def __init__(self, LOG_FILEPATH):
		self.LOG_FILEPATH = LOG_FILEPATH



		self.splitList = None

		self.epoch = []
		self.loss = []
		self.train_accuracy = []
		self.test_accuracy = []
		self.confusion = None
		self.kernel = None
		self.batch = None
		self.getKernelBatch()

		with open(self.LOG_FILEPATH, 'r') as f:
			f.readline()
		self.readLine()

	def getKernelBatch(self):
		kernel_index = self.LOG_FILEPATH.find('K')
		end_index = self.LOG_FILEPATH.find('.')
		kernelbatch = self.LOG_FILEPATH[kernel_index: end_index].split('_')
		self.kernel = kernelbatch[0][1:]
		self.batch = kernelbatch[1][1:]

	def readLine(self):
		not_EOF = True
		with open(self.LOG_FILEPATH, 'r') as f:
			f.readline()

			while not_EOF:
				line = f.readline()
				if line == '':
					con = self.splitlist[1][:-4]
					consplit = con.split('], [')
					self.confusion = np.fromstring(consplit[0], sep = ',')
					for i in range (1, len(consplit)):
						self.confusion = np.concatenate((self.confusion, np.fromstring(consplit[i], sep = ',')))
					dim = int(math.sqrt(self.confusion.shape[0]))
					self.confusion = self.confusion.reshape((dim, dim))

					not_EOF = False

				else:
					self.splitlist = line.split('[[')
					data = np.fromstring(self.splitlist[0][1:-2], sep=',')
					self.epoch.append(data[0])
					self.loss.append(data[1])
					self.train_accuracy.append(data[2])
					self.test_accuracy.append(data[3])

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


class OverallWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QHBoxLayout(self)

		self.leftwidget = LeftSideWidget(self, file)
		self.rightwidget = RightSideWidget(self, file)

		self.layout.addWidget(self.leftwidget)
		self.layout.addWidget(self.rightwidget)

class LeftSideWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)
		
		self.losswidget = PlotLoss(self, file)
		self.accuracywidget = PlotAccuracy(self, file)

		self.layout.addWidget(self.losswidget)
		self.layout.addWidget(self.accuracywidget)


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


class RightSideWidget(QWidget):
	def __init__(self, parent, file):

		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)
		
		self.textwidget = QLabel()
		self.textwidget.setText('  Kernel Size is ' + file.getKernel() + '\n  Batch size is ' + file.getBatch())
		self.textwidget.setFont(QFont('Arial', 20)) 
		self.textwidget.resize(200, 150)

		self.confusionwidget = PlotConfusion(self, file)

		self.layout.addWidget(self.textwidget)
		self.layout.addWidget(self.confusionwidget)


class PlotConfusion(QWidget):
	def __init__(self, parent, file):

		axis_labels = ['AMADEC', 'ARRAUR','CORALT','DYSMEN', 'EUPIMI','HENLES','HYLDEC','LOPPIT', 'TANGYR', 'TANICT']

		super(QWidget, self).__init__(parent)
		self.figure = plt.figure(figsize=(10,5))
		self.resize(400,400)

		self.canvas = FigureCanvas(self.figure)

		ax = self.figure.add_subplot(111)
		ax.set_title('Confusion Matrix on Last Epoch')
		ax = sns.heatmap(file.getConfusion(), xticklabels = axis_labels, yticklabels = axis_labels)

		ax.tick_params(axis = 'x', labelrotation = 90)

		self.canvas.draw()
		
		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		self.setLayout(layout)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	x = App(LOG_FILEPATH)
	sys.exit(app.exec_())



