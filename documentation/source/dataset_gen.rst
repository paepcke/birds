data_augmentation/src/create_dataset.sh
========================================================

Downloading, processing and creating a training dataset
--------------------------------------------------------
NOTE: Make sure your environment is setup per the :ref:`setup-label` instructions.

**Using the default dataset**

This process will create a dataset using the specifc files, species, and processes set by default in create_dataset.sh. By default, this code will create a dataset with 20 classes representing the song and call of 10 distinct species. The samples are from zeno-canto and are filtered with a bandpass. Additionally, for ever original sample, a second modified sample will be generated and added to the dataset. These modified samples will be randomly timeshifted (with the data "rolled") and have their volume randomly adjust by a moderate amount. All of these samples are then conmverted into spectrograms and then randmoly split into a training set and a validation set.

1. In your terminal, navigate to "data_augmentation/src".

2. Pick a name for this dataset.

2. Run the command "mkdir 'your dataset name'" in the terminal.

4. Use your preffered text editor to open create_dataset.sh.

5. Set "out_path" to the name of the dataset to be created.

6. Run the command "./create_dataset.sh".

7. Wait for the program to complete. This may take several hours.

8. Follow the steps in :ref:`train&eval_label`.

**Creating a custom dataset**

1. If you want to change which samples are used follow steps 2 through 7, otherwise skip to step 8.

2. In your terminal, navigate to "src/".

3. Use your preffered text editor to open and edit recording_download.py.

4. Read the documentation for the file and change the code to download other samples from xeno-canto or another website with an API.

5. Navigate to "data_augmentation/src".

6. Use your preffered text editor to open and edit create_dataset.sh.

7. Edit all lines that refer to samples by name (Ex: "mv *Cathar* ../") to use the names of your samples and you selected species.

8. If you want to change how samples are processed you can change what code is called on lines 61-74 in create_dataset.sh and what the arguements are. To learn what the code run by each of those lines does, read the documentation for that specific code.

9. If you want to make more signficant changes to the processing of samples, you can edit the files refferences in lines 61-74 or call your own code.

10. To change how the data is split between trianing and validation, read :ref:`train&eval_label` and the documentation for CreateTrainValidation.py and manager.py.