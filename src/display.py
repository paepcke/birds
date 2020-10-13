import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import random
import csv
import pandas as pd
import seaborn as sns
import json
import math

LOG_FILEPATH = '/Users/LeoGl/Documents/bird/09-10-2020_11-58_K7_B32.jsonl'
# LOG_FILEPATH = '/Users/LeoGl/Documents/bird/fullAugmentedSongAndCall_K7_B32.jsonl'
# LOG_FILEPATH = '/Users/LeoGl/Documents/bird/logs/05-09-2020_18-17_K7_B128.jsonl'


class App(QMainWindow):
	def __init__(self, LOG_FILEPATH):
		super().__init__()
		# initialize the file reader
		self.file = FileRead(LOG_FILEPATH)

		self.title = 'Birdsong Classifier'
		self.left = 0
		self.top = 0
		self.width = 3000
		self.height = 2000

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
		self.normal = None
		self.kernel = None
		self.batch = None
		self.files = [[[]]*20]*20
		self.getKernelBatch()

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

					if len(self.splitlist) > 2:
						# parse the file names
						one_dim_list = self.splitlist[2][:-6].split(',\"')
						item = 0
						# create a list of file names for each cell in the matrix
						for z in range(dim):
							for y in range(dim):
								# split the string into a list of files
								file_list = one_dim_list[item].split(',')[-1:]
								# remove parts of the filepath that is not the file name
								# for n in range(len(file_list)):
								# 	file_list[n] = file_list[n].split('/')[9]
								# add each list of files to the files array
								self.files[z][y] = file_list
								item += 1

					not_EOF = False

				else:
					self.splitlist = line.split('[[')
					data = np.fromstring(self.splitlist[0][1:-2], sep=',')
					self.epoch.append(data[0])
					self.loss.append(data[1])
					self.train_accuracy.append(data[2])
					self.test_accuracy.append(data[3])
		self.genTru()
		self.calcNorm()

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

	def calcNorm(self):
		self.normal = self.confusion.copy()
		self.normal[0:len(self.normal)-1][0:len(self.normal[0])-1] = 0
		for pred in range(len(self.confusion)):
			for truth in range(len(self.confusion[0])):
				if self.tru[pred][pred] != 0:
					self.normal[pred][truth] = self.confusion[pred][truth] / self.tru[pred][pred]

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

		# axis_labels = ['AMADEC', 'ARRAUR','CORALT','DYSMEN', 'EUPIMI','HENLES','HYLDEC','LOPPIT', 'TANGYR', 'TANICT']
		self.axis_labels = ['AMADEC_CALL', 'AMADEC_SONG', 'ARRAUR_CALL', 'ARRAUR_SONG', 'CORALT_CALL', 'CORALT_SONG', 'DYSMEN_CALL_SONG', 'DYSMEN_SONG', 'EUPIMI_CALL',
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
		self.cid = self.canvas.mpl_connect("button_release_event", self.onRelease)
		self.cid = self.canvas.mpl_connect("motion_notify_event", self.onMotion)
		self.heatmap = ax

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

	def onRelease(self, event):
		if self.xint != -1 and self.yint != -1:
			print("actual species:", self.axis_labels[self.xint])
			print("predicted species:", self.axis_labels[self.yint])
			print("normalized accuracy:", self.file.getNormalized()[self.yint][self.xint])
			file_names = self.file.getFileMatrix()[self.yint][self.xint]
			if file_names != []:
				print("files names:", file_names)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	x = App(LOG_FILEPATH)
	sys.exit(app.exec_())
