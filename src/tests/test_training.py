'''
Created on Dec 6, 2020

@author: paepcke
'''
import glob
import os
from pathlib import Path
import unittest

from training import EPOCHS, SEED, BATCH_SIZE, KERNEL_SIZE
from training import Training
import training


#********
#********
#******TEST_ALL = True
TEST_ALL = False

CURR_DIR = os.path.join(os.path.dirname(__file__))
FILEPATH = os.path.join(CURR_DIR, 'data/')
training.FILEPATH = FILEPATH
NET_NAME = 'Resnet18Partial'



class TestTraining(unittest.TestCase):

    GPU_INDEX = 0

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):

        # How many training and validation .png samples 
        # do we have in the test dir?
        
        train_root         = os.path.join(FILEPATH, 'train')
        validate_root      = os.path.join(FILEPATH, 'validate')
        
        # Number of classes is the number of subdirectories
        # under the train (or validate) directory:
        cls.num_classes    = len(os.listdir(train_root))
        
        cls.train_files    = [file_path 
                                for file_path 
                                 in glob.iglob(train_root + '**/**', recursive=True)
                                 if os.path.isfile(file_path)
                                 ]
        cls.validate_files = [file_path 
                                for file_path 
                                 in glob.iglob(validate_root + '**/**', recursive=True)
                                 if os.path.isfile(file_path)
                                 ] 

        cls.num_train_samples = len(cls.train_files)
        cls.num_validate_samples = len(cls.validate_files)
        
    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        # file_path, epochs, batch_size, kernel_size, seed=42, net_name='BasicNet', gpu=0):
        self.training = Training(FILEPATH,
                                 EPOCHS,
                                 BATCH_SIZE,
                                 KERNEL_SIZE,
                                 SEED,
                                 'Resnet18Partial',
                                 self.GPU_INDEX,
                                 unit_testing=True)
        self.using_gpu = self.training.device.type.startswith('cuda')

    #------------------------------------
    # tearDown 
    #-------------------


    def tearDown(self):
        pass

    #------------------------------------
    # tearDownClass 
    #-------------------
    
    @classmethod
    def tearDownClass(cls):
        # Remove the .jsonl result files created during
        # training:
        for res_file in Path(CURR_DIR).glob('*.jsonl'):
            res_file.unlink()

    
    #------------------------------------
    # testGetNet 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testGetNet(self):
        model = self.training.get_net('Resnik18', 
                                      BATCH_SIZE, 
                                      KERNEL_SIZE, 
                                      self.GPU_INDEX
                                      )
        self.assertEqual(model._get_name(), 'ResNet')


    #------------------------------------
    # testImportData 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testImportData(self):
        (train_data_loader, test_data_loader) = self.training.import_data()
        self.assertListEqual(train_data_loader.dataset.classes,
                             ['DYSMEN_S', 'HENLES_S'])
        self.assertDictEqual(train_data_loader.dataset.class_to_idx,
                             {'DYSMEN_S': 0, 'HENLES_S': 1})
        self.assertEqual(len(train_data_loader.dataset), 12)
        
        self.assertListEqual(test_data_loader.dataset.classes,
                             ['DYSMEN_S', 'HENLES_S'])
        self.assertDictEqual(test_data_loader.dataset.class_to_idx,
                             {'DYSMEN_S': 0, 'HENLES_S': 1})
        self.assertEqual(len(test_data_loader.dataset), 6)
        
    #------------------------------------
    # testDataLoader
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testDataLoader(self):
        
        # Get training data loader with default (32) number of samples per batch:
        (train_data_loader, _test_data_loader) = self.training.import_data()
        
        # Get a generator from the data loader: 
        gen   = iter(train_data_loader)
        # Get first batch:
        batches = next(gen)
        # Batches are delivered in a list. We only asked for
        # one batch, so we get a single-element list:
        batch = batches[0]
        
        # Batch is a tensor with a datapoint for every pixel. Example:
        #
        #   batch-size-num-of-images x RxGxB x img_pixel_height x img_pixel_width
        #
        # We have 12 images in the training set; default batch size is 32,
        # so all images should be in this one and only batch:
        #
        # Such as [12,3,400,400] for 12 training samples, RGB, and
        # sample image dims of 400x400:

        self.assertTupleEqual(batch.shape, (self.num_train_samples,
                                            3,  # One dim each for RGB,
                                            training.SAMPLE_HEIGHT, 
                                            training.SAMPLE_WIDTH))
        
        # Now try batch size of 4, which should give us 3
        # batches of 4 samples in each batch:
        
        self.training = Training(FILEPATH,
                                 EPOCHS,
                                 4,             # Batch size
                                 KERNEL_SIZE,
                                 SEED,
                                 'Resnet18Partial',
                                 self.GPU_INDEX,
                                 testing=True)

        (train_data_loader, _test_data_loader) = self.training.import_data()
        # Get a generator from the data loader: 
        gen   = iter(train_data_loader)
        # Get first batch:
        batches = next(gen)
        # Batches are delivered in a list. We only asked for
        # one batch, so we get a single-element list:
        batch = batches[0]
        
        self.assertTupleEqual(batch.shape, (4,  # Num of sample images in one batch 
                                            3,  # One dim each for RGB,
                                            training.SAMPLE_HEIGHT, 
                                            training.SAMPLE_WIDTH))

        # We should have two more batches (3 batches of 4 each,
        # and we pulled one above):
        remains = [batch for batch in gen]
        self.assertEqual(len(remains), 2)


    #------------------------------------
    # testTrain 
    #-------------------

    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testTrain(self):
        
        # Create a fully functional Training instance
        # by setting 'testing' to False:
        
        # 3 batches of 4 samples in each batch:
        self.training = Training(FILEPATH,
                                 EPOCHS,
                                 4,             # Batch size
                                 KERNEL_SIZE,
                                 SEED,
                                 'Resnet18Partial',
                                 self.GPU_INDEX,
                                 unit_testing=False)
        self.training.train()


# -------------------- Main --------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()