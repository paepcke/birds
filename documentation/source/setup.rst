.. _setup-label:

Setup Instructions
========================================================

**1.** Download the code from https://github.com/paepcke/birds.

**2.** Navigate to the src folder in files you download.

**3.** Use python to run the setup.py file (type "python setup.py install" in your terminal)

**4.** If you are using a linux computer, run the command "sudo apt-get install libsndfile1" in your terminal.

**5.** Navigate to the "src" folder in your terminal and run "python recording_download.py 'name of the folder to put the samples'". For more info on this step, go to :ref:`download_label`.

**6.** If you plan to, use :ref:`manager_label` to create and train a group of models you are DONE. Otherwise, ignore this step, and move on to step 7.

**7.** Create a "train" folder and a "validation" folder in the SAME directory that the PARENT directory for the samples is in.

**8.** Navigate to "data_augmentation/src/" in your terminal and run "python CreateTrainValidation 'name of the parent directory that the samples are in'". The parameter you enter should be the same as the one in step 4.

**9.** Use either :ref:`manager_label` or :ref:`training_label` to create and train the model. To learn more go to :ref:`train&eval_label`.