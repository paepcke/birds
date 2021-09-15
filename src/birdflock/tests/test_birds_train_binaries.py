'''
Created on Sep 11, 2021

@author: paepcke
'''
import os
#from sched import scheduler
import unittest

from birdflock.birds_train_binaries import BinaryBirdsTrainer
from data_augmentation.multiprocess_runner import Task
import multiprocessing as mp


TEST_ALL = True
#TEST_ALL = False


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.snippet_root = os.path.join(cls.cur_dir, 'data/snippets')
        cls.species1     = 'VASEG'
        cls.species2     = 'PATYG'
        cls.species3     = 'YOFLG'


    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------------- Tests ---------------

    #------------------------------------
    # test_constructor
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        trainer = BinaryBirdsTrainer(self.snippet_root)
        
        self.assertEqual(len(trainer.tasks_to_run), 3)
        
        trainer.train()
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
        trainer = BinaryBirdsTrainer(self.snippet_root,
                                     species_list=species_to_train
                                     )
        self.assertEqual(trainer.tasks, species_to_train)

        trainer.train()


    #------------------------------------
    # test_await_any_job_done
    #-------------------

    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test__await_any_job_done(self):
    #
    #     func_runner = scheduler()
    #
    #     trainer = BinaryBirdsTrainer(self.snippet_root)
    #
    #     task1,task2,task3 = trainer.tasks_to_run
    #
    #     for i, task in enumerate(trainer.tasks_to_run):
    #         task.done_event = mp.Event()
    #         task.shared_return_dict = {'result' : f"task_{i}"}
    #
    #     func_runner.enter(3, 1, lambda task: task.done_event.set(), argument=(task1,))
    #     func_runner.run()
    #
    #     print("Fake-waiting for task1...")
    #     trainer.train(unittesting=True)
    #     print("Done fake-waiting for task1")
    #
    #     self.assertTrue(task1.done_event.is_set())
    #
    #     func_runner.enter(3, 1, lambda task: task.done_event.set(), argument=(task2,))
    #     print("Fake-waiting for task2...")
    #     func_runner.run()
    #     print("Done fake-waiting for task2")
    #
    #     self.assertTrue(task2.done_event.is_set())
    #
    #     func_runner.enter(3, 1, lambda task: task.done_event.set(), argument=(task3,))
    #     print("Fake-waiting for task3...")
    #     func_runner.run()
    #     print("Done fake-waiting for task3")
    #
    #     self.assertTrue(task3.done_event.is_set())
    #
    #     print('foo')

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