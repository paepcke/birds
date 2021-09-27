'''
Created on Sep 11, 2021

@author: paepcke
'''
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiment_manager.experiment_manager import ExperimentManager, Datatype

from birdflock.birds_train_binaries import BinaryBirdsTrainer
from data_augmentation.multiprocess_runner import Task
from data_augmentation.utils import Utils
import multiprocessing as mp


TEST_ALL = True
TEST_ALL = False


class BirdsBinaryTrainerTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.snippet_root = os.path.join(cls.cur_dir, 'data/snippets')
        cls.species1     = 'VASEG'
        cls.species2     = 'PATYG'
        cls.species3     = 'YOFLG'
        
        # Read the standard config file, and 
        # make some changes to match this testing
        # context:
        proj_root = Path(cls.cur_dir).parent.parent.parent
        config_path = proj_root.joinpath('config.cfg')
        cls.config  = Utils.read_configuration(str(config_path))
        # Change the species root to this test suite's data
        cls.config['Paths']['root_train_test_data'] = cls.snippet_root

    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory(dir='/tmp', prefix='binTrainTst')

    def tearDown(self):
        self.tmpdir_obj.cleanup()

# -------------------- Tests ---------------

    #------------------------------------
    # test_constructor
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        trainer = BinaryBirdsTrainer(self.config,
                                     experiment_path=self.tmpdir_obj.name
                                     )
        
        self.assertEqual(len(trainer.tasks_to_run), 3)
        
        trainer.train()
        for task in trainer.tasks_to_run:
            self.assertIsNone(task.error)
        print('Nothing was tested, but fit() finished')
        
        
    #------------------------------------
    # test_callback_scoring
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_callback_scoring(self):
        # To be added; or not.
        pass
    
    #------------------------------------
    # test_supplying_species_list
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_supplying_species_list(self):
        species_to_train = ['VASEG', 'YOFLG']
        trainer = BinaryBirdsTrainer(self.config,
                                     focals_list=species_to_train,
                                     experiment_path=self.tmpdir_obj.name
                                     )
        self.assertEqual(trainer.tasks, species_to_train)
        
        trainer.train()
        
        for task in trainer.tasks_to_run:
            self.assertIsNone(task.error)

    #------------------------------------
    # test_supplying_timestamp
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_supplying_timestamp(self):
        species_to_train = ['VASEG', 'YOFLG']
        tst_timestamp = '2019-09-30T12_23_57'
        
        # Test the timestamp format sanity check
        try:
            trainer = BinaryBirdsTrainer(self.config,
                                         focals_list=species_to_train,
                                         timestamp='foo-bar',
                                         experiment_path=self.tmpdir_obj.name
                                         )
            raise AssertionError("BinaryBirdsTrainer failed to recognize bad timestamp")
        except ValueError:
            # Good:
            pass
        
        trainer = BinaryBirdsTrainer(self.config,
                                     focals_list=species_to_train,
                                     timestamp=tst_timestamp,
                                     experiment_path=self.tmpdir_obj.name
                                     )
        
        trainer.train()
        
        # There should be two experiments with
        # a 2019 time stamp
        
        exp_root = trainer.experiment_path
        found_dirs = set([])
        for file_or_dir in Utils.listdir_abs(exp_root):
            if not os.path.isdir(file_or_dir):
                continue
            timestamp = Utils.timestamp_from_exp_path(file_or_dir)
            if timestamp != tst_timestamp:
                continue
            else:
                found_dirs.add(Path(file_or_dir).stem)
                
        expected = set(['Classifier_YOFLG_2019-09-30T12_23_57',
                        'Classifier_VASEG_2019-09-30T12_23_57'
                        ])
        
        self.assertSetEqual(found_dirs, expected)
        print(trainer)

    #------------------------------------
    # test_save_classifier_history
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_save_classifier_history(self):
        species_to_train = ['VASEG', 'YOFLG']
        tst_timestamp = '2019-09-30T12_23_57'
        root_dir_nm   = self.tmpdir_obj.name
        trainer = BinaryBirdsTrainer(self.config,
                                     focals_list=species_to_train,
                                     timestamp=tst_timestamp,
                                     experiment_path=root_dir_nm
                                     )
        
        trainer.train()
        yoflg_exp = ExperimentManager(os.path.join(root_dir_nm, 
                                                   'Classifier_YOFLG_2019-09-30T12_23_57'))
        vaseg_exp = ExperimentManager(os.path.join(root_dir_nm, 
                                                   'Classifier_VASEG_2019-09-30T12_23_57'))

        yoflg_df = yoflg_exp.read('YOFLG_res_by_epoch', Datatype.tabular)
        vaseg_df = vaseg_exp.read('VASEG_res_by_epoch', Datatype.tabular)
        
        expected_cols = pd.Index(
            ['species', 'train_loss', 'train_loss_best', 'valid_loss',
             'valid_loss_best', 'valid_acc', 'valid_acc_best', 'balanced_accuracy',
             'balanced_accuracy_best', 'f1', 'f1_best', 'accuracy', 'accuracy_best']
            )
        
        # Outcomes are a bit different dur 
        # to random data selection. So just
        # check that the result data frames are
        # retrievable and have the expected columns
        self.assertTrue((yoflg_df.columns == expected_cols).all())
        self.assertTrue((vaseg_df.columns == expected_cols).all())

    #------------------------------------
    # test_early_stopping
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_early_stopping(self):
        species_to_train = ['VASEG', 'YOFLG']
        tst_timestamp = '2019-09-30T12_23_57'
        root_dir_nm   = self.tmpdir_obj.name
        trainer = BinaryBirdsTrainer(self.config,
                                     focals_list=species_to_train,
                                     timestamp=tst_timestamp,
                                     experiment_path=root_dir_nm
                                     )
        
        trainer.train()

# ---------------------- Utilities ----------------

    def make_task(self, name):
        
        task_evnt = mp.Event()
        task = Task('name',
                    None, # Target function
                    shared_return_dict = {name : f"task_{name}"},
                    done_event=task_evnt
                    )
        return task

# -------------------- Main ---------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    