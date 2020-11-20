.. _train&eval_label:

Training and Results
========================================================

Model Training
--------------------------------------------------------
NOTE: Make sure your environment is setup per the :ref:`setup-label` instructions.

**Training a single model**

This process will train a single model with a single split of data between trianing and evaluation. This will generate a single log file that can be identified by the time and data that the following commands were run.

1. In your terminal, navigate to "src/".

2. If you want to use the default parameters then simply run the command "python training.py 'filepath to the samples'".

3. Otherwise, use your preferred text-editor to open training.py.

4. Modify the variables at the top of the file in all CAPS to the values you want them to be set to. To learn what these values do, go to :ref:`training_label`.

5. Save and quit training.py

6. run the command "python training.py".

**Training a set of models**

This process uses a Stratified K-Split and Cross validation to train a series of models with different splits of the data. Each model will have its own log file, identified by a time and data that the model was trained.

1. In your terminal, navigate to "src/".

2. If you want to use the default parameters then simply run the command "python manager.py 'filepath to the samples'".

3. Otherwise, use your preferred text-editor to open manager.py.

4. Modify the variables at the top of the file in all CAPS to the values you want them to be set to. To learn what these values do, go to :ref:`manager_label`.

5. Save and quit manager.py

6. run the command "python manager.py".

Displaying Results
--------------------------------------------------------
NOTE: Make sure your environment is setup per the :ref:`setup-label` instructions.

**Running the program**

1. In your terminal, navigate to "src/".

2. Run the command "python display.py 'filepath to the log file'".

**Understanding the results**

 * On the top left, is a graph showing loss vs epoch.

 * On the top right are some statistics about the model.

 * On the bottom left, is a graph of MAP, testing accuracy, and training accuracy vs epoch.

 * On the bottom right is a confusion matrix for the final epoch the model was trained for. Click on any cell in the confusion matrix to get a list of the samples that were misclassified by the model in that cell. Many cells have none.
