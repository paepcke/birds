import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Evaluations:

    # return the confusion matrix, followed by precision and recall
    def compute_cm(model, data_loader):
        """
        Performs preliminary data analysis of the testing DataLoader.
        :param data_loader: the test_data_loader to gather information from
        :type data_loader: DataLoader
        :returns: a confusion matrix, precision and recall values, and the paths of misclassified samples
        """
        test_labels = []
        test_predictions = []
        # finds the number of correctly labelled samples
        for batch_images, batch_labels in data_loader:
            outputs = model(batch_images)
            _, predictions = torch.max(outputs.data, 1)
            # Add batch labels and predictions to total list
            test_labels.extend(list(np.numpy(batch_labels)))
            test_predictions.extend(list(np.numpy(predictions)))

        # add misclassified sample paths to a list
        # incorrect_paths = [[[] for i in range(20)] for j in range(20)]
        # for i in range(len(labels) - 1):
        #     if predicted[i] != labels[i]:
        #         path[i] = path[i].split('/')[len(path[i].split('/')) - 1]
        #         incorrect_paths[predicted[i]][labels[i]].append(path[i])

        # Calculate the confusion matrix.
       cm = confusion_matrix(test_labels, test_predictions)
       # Log the confusion matrix as an image summary.
       class_name = dataloader.class_to_idx.keys()
       cm_figure = plot_confusion_matrix(cm, class_names=class_names)
        # calculate precision and recall
        precision, recall = [], []
        for i in range (0, len(cm)):
            if sum(confusion[i]) != 0:
                precision.append(cm[i][i] / sum(cm[i]))
            else:
                precision.append(0.0)
            recall.append(cm[i][i] / sum(cm[:,i]))

        return cm, cm_figure, precision, recall#, incorrect_paths

# Below from tensorflow.org
def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure
