"""
File Name: display.py
Authors: Leo Glikbarg, Amy Dunphy
Owner: Stanford Center for Conservation Biology

Displays the training results as recorded in a .jsonl log file.
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import seaborn as sns
import math
import warnings

# the absolute or relative filepath to the log file to display
LOG_FILEPATH = '/Users/LeoGl/Documents/bird/02-11-2020_13-08_K7_B32.jsonl'

# The default dimensions of the entire window
WIDTH = 1500
HEIGHT = 1000


class App(QMainWindow):
	"""
	QT Window class which contains all QT widgets
	"""

	def __init__(self, log_filepath):
		"""
		:param log_filepath: the absolute or relative path of the log file to be displayed
		"""
		super().__init__()
		# initialize the file reader
		self.file = FileRead(LOG_FILEPATH)

		# sets the size and shape of the window
		self.title = 'Birdsong Classifier'
		self.left = 0
		self.top = 0
		self.width = WIDTH
		self.height = HEIGHT

		# add in widgets
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.allwidget = OverallWidget(self, self.file)

		self.setCentralWidget(self.allwidget)
		self.show()


class FileRead:
	"""
	Parses the log file, and stores data

	Parses the file line by line, stores relevant info, runs basic data analysis of info (ie. MAP) and has methods to
	fetch info.
	"""

	def __init__(self, log_filepath):
		"""
		:param log_filepath: the absolute or relative path of the log file to be displayed
		"""
		self.LOG_FILEPATH = log_filepath

		self.splitList = None

		self.epoch = []
		self.loss = []
		self.train_accuracy = []
		self.test_accuracy = []
		self.confusion = None
		self.normal = None
		self.tru = None
		self.kernel = None
		self.batch = None
		self.MAP = None
		self.MAPs = []
		# create empty tensor to store the names of misclassified files in
		self.files = [[[] for i in range(20)] for j in range(20)]
		self.getKernelBatch()

		self.readLine()

	def getKernelBatch(self):
		"""
		Get the kernel and batch size of the model

		parses the log file and searches for the kernel and batch size which are then set in the appropriate fields.
		"""
		kernel_index = self.LOG_FILEPATH.find('K')
		end_index = self.LOG_FILEPATH.find('.')
		kernelbatch = self.LOG_FILEPATH[kernel_index: end_index].split('_')
		self.kernel = kernelbatch[0][1:]
		self.batch = kernelbatch[1][1:]

	def readLine(self):
		"""
		Parses the lof file and collects are important info

		Parses the file line by line, and recording the confusion matrix, accuracies, loss, and misclassified sample
		file names. Sets the attributes of the the FileRead() class to their appropriate values.
		"""
		not_EOF = True
		with open(self.LOG_FILEPATH, 'r') as f:
			f.readline()
			confusion = None

			while not_EOF:
				line = f.readline()
				# parse just the last epoch
				if line == '':
					# record confusion matrix for last epoch
					self.confusion = confusion

					# parse and store names of misclassified files in final epoch
					one_dim_list = self.splitList[1:21]
					x = 0
					for row in one_dim_list:
						y = 0
						for cell in row.split('], ['):
							self.files[x][y].append(cell)
							y += 1
						x += 1
					not_EOF = False

				# for every epoch that is not the last one
				else:
					# create list of info about epoch
					self.splitList = line.split('[[')
					data = np.fromstring(self.splitList[0][1:-2], sep=',')
					# parse and store epoch, loss, and accuracies for current epoch
					self.epoch.append(data[0])
					self.loss.append(data[1])
					self.train_accuracy.append(data[2])
					self.test_accuracy.append(data[3])

					# parse and calculate the confusion matrix for the current epoch
					con = self.splitList[21][:-4]
					consplit = con.split('], [')
					confusion = np.fromstring(consplit[0], sep=',')
					for i in range(1, len(consplit)):
						confusion = np.concatenate((confusion, np.fromstring(consplit[i], sep=',')))
					dim = int(math.sqrt(confusion.shape[0]))
					confusion = confusion.reshape((dim, dim))
					# calculate and store the MAP for the current epoch
					self.calcMAP(confusion)
		# run analysis on data parsed
		self.genTru()
		self.calcNorm()
		self.calcMAP()

	def genTru(self):
		"""
		Calculates a confusion matrix with only correct classifications

		Removes all cells from the confusion matrix of the last epoch where the sample is misclassified.
		"""
		# move all misclassified samples to correct location, zeros out other cells
		self.tru = self.confusion.copy()
		for pred in range(len(self.confusion)):
			for truth in range(len(self.confusion[0])):
				if truth != pred:
					self.tru[truth][truth] += self.confusion[pred, truth]
					self.tru[pred, truth] = 0

	def calcNorm(self):
		"""
		Calculates a normalized confusion matrix

		Normalizes the confusion matrix for the last epoch by the number of samples each species has.
		"""
		self.normal = self.confusion.copy()
		self.normal[0:len(self.normal) - 1][0:len(self.normal[0]) - 1] = 0
		for pred in range(len(self.confusion)):
			for truth in range(len(self.confusion[0])):
				if self.tru[pred][pred] != 0:
					self.normal[pred][truth] = self.confusion[pred][truth] / self.tru[pred][pred]

	def calcMAP(self, confusion=None):
		"""
		Calculates the mean average precision (MAP) of a confusion matrix.

		:param confusion: a confusion matrix to calculate the MAP of. matrix of last epoch used if None
		"""
		if confusion is None:
			confusion = self.confusion
		# create a copy with correct classifications zeroed out
		temp = confusion.copy()
		for pred in range(len(confusion)):
			for truth in range(len(confusion[0])):
				if pred == truth:
					temp[pred][truth] = 0
		# create an list of the number of false positives by species
		false_pos = temp.sum(axis=0)
		false_positives = np.zeros(int(len(false_pos) / 2))
		# merges song and call entries by species
		count = 0
		for species in range(len(false_pos)):
			if species % 2 == 0:
				false_positives[count] = false_pos[species]
				count += 1
		count = 0
		# add false positives to false_positives list
		for species in range(len(false_pos)):
			if species % 2 != 0:
				false_positives[count] += false_pos[species]
				count += 1
		temp = confusion.copy()
		# zero out any misclassified samples
		for pred in range(len(confusion)):
			for truth in range(len(confusion[0])):
				if pred != truth:
					temp[pred][truth] = 0
		# create an list of the number of true positives by species
		true_pos = temp.sum(axis=0)
		true_positives = np.zeros(int(len(true_pos) / 2))
		count = 0
		# merges song and call entries by species
		for species in range(len(true_pos)):
			if species % 2 == 0:
				true_positives[count] = true_pos[species]
				count += 1
		count = 0
		# add true positives to true_positives list
		for species in range(len(true_pos)):
			if species % 2 != 0:
				true_positives[count] += true_pos[species]
				count += 1
		# calculate denominator for calculation and divide
		denom = np.add(true_positives, false_positives)
		y_scores = np.true_divide(true_positives, denom)

		# move results to correct fields
		if confusion is self.confusion:
			self.MAP = np.sum(y_scores) / len(y_scores)
		else:
			self.MAPs.append((np.sum(y_scores) / len(y_scores)) * 100)

	def getMAP(self):
		"""returns MAP of last epoch"""
		return self.MAPs[len(self.MAPs) - 1]

	def getMAPs(self):
		"""returns list of MAP for each epoch"""
		return self.MAPs

	def getEpoch(self):
		"""returns list of epochs"""
		return self.epoch

	def getLoss(self):
		"""returns list of loss for each epoch"""
		return self.loss

	def getTrainingAccuracy(self):
		"""returns list of training accuracy for each epoch"""
		return self.train_accuracy

	def getTestingAccuracy(self):
		"""returns list of testing accuracy for each epoch"""
		return self.test_accuracy

	def getConfusion(self):
		"""returns confusion matrix of last epoch"""
		return self.confusion

	def getKernel(self):
		"""returns kernel size of model"""
		return self.kernel

	def getBatch(self):
		"""returns batch size of model"""
		return self.batch

	def getNormalized(self):
		"""returns normalized confusion matrix of last epoch"""
		return self.normal

	def getFileMatrix(self):
		"""returns tensor of file names of misclassified files sorted by predicted and labelled species"""
		return self.files


class OverallWidget(QWidget):
	"""
	The Overall Widget for the window

	Wrapper/container for all the other widget in the window.
	"""

	def __init__(self, parent, file):
		"""
		Parameters
		----------
		parent : parent widget, optional
			A parent widget of this widget
		file : FileRead object
			A parsed file's data
		"""
		super(QWidget, self).__init__(parent)
		self.layout = QHBoxLayout(self)

		# add the left and right widgets
		self.leftwidget = LeftSideWidget(self, file)
		self.rightwidget = RightSideWidget(self, file)

		self.layout.addWidget(self.leftwidget)
		self.layout.addWidget(self.rightwidget)


class LeftSideWidget(QWidget):
	"""
	Widget for left-side of window

	Creates two graphs that together show loss, accuracy, and MAP vs epoch.
	"""

	def __init__(self, parent, file):
		"""
		Parameters
		----------
		parent : parent widget, optional
			A parent widget of this widget
		file : FileRead object
			A parsed file's data
		"""
		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

		self.losswidget = PlotLoss(self, file)
		self.accuracywidget = PlotAccuracy(self, file)

		self.layout.addWidget(self.losswidget)
		self.layout.addWidget(self.accuracywidget)


class PlotLoss(QWidget):
	"""
	Plot of Loss vs Epoch

	Creates a widget that is a graph of loss vs epoch.
	"""

	def __init__(self, parent, file):
		"""
		Parameters
		----------
		parent : parent widget, optional
			A parent widget of this widget
		file : FileRead object
			A parsed file's data
		"""
		super(QWidget, self).__init__(parent)
		# set figure size, widget size
		self.figure = plt.figure(figsize=(10, 5))
		self.resize(300, 300)

		# create the plot, set the axis and data
		self.canvas = FigureCanvas(self.figure)
		ax = self.figure.add_subplot(111)
		ax.set_title('Loss vs Epoch')
		ax.set_ylim(0, 10)
		ax.plot(file.getEpoch(), file.getLoss(), 'r')
		self.canvas.draw()

		# layout plot in a nice way
		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		self.setLayout(layout)


class PlotAccuracy(QWidget):
	"""
	Plot of Accuracy and MAP vs Epoch

	Creates a widget that is a graph of training accuracy, testing accuracy, and MAP vs epoch.
	"""

	def __init__(self, parent, file):
		"""
		Parameters
		----------
		parent : parent widget, optional
			A parent widget of this widget
		file : FileRead object
			A parsed file's data
		"""
		super(QWidget, self).__init__(parent)
		# set figure size, widget size
		self.figure = plt.figure(figsize=(10, 5))
		self.resize(300, 300)

		self.canvas = FigureCanvas(self.figure)
		# create the plot, set the axis and data
		ax = self.figure.add_subplot(111)
		ax.set_title('Testing & Training Accuracy, and MAP vs Epoch')
		ax.set_ylim(0, 100)
		ax.plot(file.getEpoch(), file.getTrainingAccuracy(), 'b')
		ax.plot(file.getEpoch(), file.getTestingAccuracy(), 'r')
		ax.plot(file.getEpoch(), file.getMAPs(), 'c')
		self.canvas.draw()

		# layout plot in a nice way
		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		self.setLayout(layout)


class RightSideWidget(QWidget):
	"""
	Widget for right-side of window

	Creates labels with info about the model, and the training results. Graphs a normalized confusion matrix.
	"""

	def __init__(self, parent, file):
		"""
		Parameters
		----------
		parent : parent widget, optional
			A parent widget of this widget
		file : FileRead object
			A parsed file's data
		"""
		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

		# create labels and set text size
		self.file_label = QLabel()
		self.file_list = QLineEdit()
		self.textwidget = QLabel()
		self.textwidget.setFont(QFont('Arial', 20))

		# plot the confusion matrix, set label with info
		self.file = file
		self.confusionwidget = PlotConfusion(self, file)
		self.textwidget.setText('  Accuracy: ' + str(max(self.file.getTestingAccuracy()))[:7]
								+ '\n  MAP: ' + str(self.file.getMAP())[:7])

		# add in widgets, connect method to release event
		self.layout.addWidget(self.textwidget)
		self.layout.addWidget(self.confusionwidget)
		self.cid = self.confusionwidget.canvas.mpl_connect("button_release_event", self.onRelease)

	def onRelease(self, event):
		"""
		Called when a user clicks and releases somewhere on the RightSideWidget. Fetchs more detailed information
		about the model in general and more information on the cell that was clicked on. Displays species info,
		batch size, kernel size, and the file names of misclassified samples.
		"""
		# fetch info for the selected cell
		actual = self.confusionwidget.axis_labels[self.confusionwidget.xint]
		predicted = self.confusionwidget.axis_labels[self.confusionwidget.yint]
		norm_acc = self.confusionwidget.file.getNormalized()[self.confusionwidget.yint][self.confusionwidget.xint]
		file_names = self.confusionwidget.file.getFileMatrix()[self.confusionwidget.yint][self.confusionwidget.xint]
		# if there are misclassified samples for the selected cells, display them
		self.file_list.setText('none')
		if file_names:
			# add widgets to display the file names
			self.file_label.setText('File names of misclassified samples:')
			self.file_list.setText(str(file_names[0])[1:-1])
			self.file_list.setFont(QFont('Arial', 10))
			self.file_label.setFont(QFont('Arial', 10))
			self.file_label.setFixedHeight(50)
			self.layout.addWidget(self.file_label)
			self.layout.addWidget(self.file_list)
			# change the text in the upper right of the window
			self.textwidget.setText('  Accuracy: ' + str(max(self.file.getTestingAccuracy()))[:7]
									+ '\n  MAP: ' + str(self.file.getMAP())[:7] + '\n  Actual Species:  ' + str(
				actual) + '\n  Predicted Species:  ' + str(predicted) + '\n  Normalized Accuracy:  ' +
									str(norm_acc)[:7])
			self.textwidget.setFont(QFont('Arial', 10))


class PlotConfusion(QWidget):
	"""
	Confusion Matrix widget

	Sets the axis labels, the size of the figure, and the whole widget. Initializes miscellaneous values, sets labels
	and values, and then graphs the confusion matrix as a normalized heatmap. Creates a connection between a motion
	event and the OnMotion method.
	"""

	def __init__(self, parent, file):
		"""
		Parameters
		----------
		parent : parent widget, optional
			A parent widget of this widget
		file : FileRead object
			A parsed file's data
		"""

		# axis_labels = ['AMADEC', 'ARRAUR','CORALT','DYSMEN', 'EUPIMI','HENLES','HYLDEC','LOPPIT', 'TANGYR', 'TANICT']
		self.axis_labels = ['AMADEC_CALL', 'AMADEC_SONG', 'ARRAUR_CALL', 'ARRAUR_SONG', 'CORALT_CALL', 'CORALT_SONG',
							'DYSMEN_CALL', 'DYSMEN_SONG', 'EUPIMI_CALL',
							'EUPIMI_SONG', 'HENLES_CALL', 'HENLES_SONG', 'HYLDEC_CALL', 'HYLDEC_SONG', 'LOPPIT_CALL',
							'LOPPIT_SONG', 'TANGYR_CALL', 'TANGYR_SONG',
							'TANICT_CALL', 'TANICT_SONG']

		# inherits QWidget
		super(QWidget, self).__init__(parent)
		# sets sizes
		self.figure = plt.figure(figsize=(10, 5))
		self.resize(400, 400)
		self.canvas = FigureCanvas(self.figure)

		# default values
		self.xint = -1
		self.yint = -1
		self.file = file

		# create confusion matrix and its peripherals
		ax = self.figure.add_subplot(111)
		ax.set_title('Confusion Matrix on Last Epoch')
		ax = sns.heatmap(file.getNormalized(), xticklabels=self.axis_labels, yticklabels=self.axis_labels, center=0.45)
		ax.set_xlabel('actual species')
		ax.set_ylabel('predicted species')
		ax.tick_params(axis='x', labelrotation=90)

		# add all the widgets & figures, lay them out nicely
		self.canvas.draw()
		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		self.setLayout(layout)
		self.cid = self.canvas.mpl_connect("motion_notify_event", self.onMotion)
		self.rect = None
		self.heatmap = ax

	def onMotion(self, event):
		"""
		Called when a cursor moves across the confusion matrix. Tracks position of cursor and highlights the
		cell under the cursor.
		"""
		if not event.inaxes:  # if the cursor is not on a cell in the confusion matrix
			self.xint = -1
			self.yint = -1
			return

		# update which cell is selected and redraw a higlighted reactangle around that call
		self.xint = int(event.xdata)
		self.yint = int(event.ydata)
		self.rect = mpatches.Rectangle((self.xint, self.yint), 1, 1, fill=False, linestyle='dashed', edgecolor='red',
									   linewidth=2.0)

		self.heatmap.add_patch(self.rect)
		self.canvas.draw()
		self.rect.remove()


if __name__ == '__main__':
	"""creates and runs a QT app which displays information stored at LOG_FILEPATH"""
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	app = QApplication(sys.argv)
	x = App(LOG_FILEPATH)
	sys.exit(app.exec_())
