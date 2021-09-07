'''
Created on Sep 6, 2021

@author: paepcke
'''
from enum import Enum
import random
import signal
import time
import unittest

from data_augmentation.multiprocess_runner import MultiProcessRunner, Task


TEST_ALL = True
#TEST_ALL = False

class WorkerBehavior(Enum):
    BASIC           = 1
    RAISE_EXCEPTION = 2
    INFINITE_LOOP   = 3

class MultiProcessRunnerTester(unittest.TestCase):


    def setUp(self):
        self.tasks = {}


    def tearDown(self):
        pass

# ------------------ Tests -----------------

    #------------------------------------
    # test_one_task
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_one_task(self):
        
        task_name = self.make_basic_task()
        results = MultiProcessRunner(self.tasks[task_name]).results
        
        # Only one task, so only one result:
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertDictEqual(res,
                             {'arg1'  : 'arg1-task_1', 
                              'kwarg1': 'kwarg1-task_1'
        })

    #------------------------------------
    # test_two_tasks
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_two_tasks(self):

        # Two tasks:
        self.make_basic_task()
        self.make_basic_task()
        results = MultiProcessRunner(list(self.tasks.values())).results
        
        # Got something like:
        #    [{'arg1': 'arg1-task_1', 'kwarg1': 'kwarg1-task_1'}, 
        #     {'arg1': 'arg1-task_2', 'kwarg1': 'kwarg1-task_2'}
        #     ]
        self.assertEqual(len(results), 2)
        
        for res_dict in results:
            self.assertIn(res_dict['arg1'],   ['arg1-task_1', 'arg1-task_2'])
            self.assertIn(res_dict['kwarg1'], ['kwarg1-task_1', 'kwarg1-task_2'])
        
    #------------------------------------
    # test_exception_in_child
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_exception_in_child(self):

        self.make_basic_task(WorkerBehavior.RAISE_EXCEPTION)
        
        results = MultiProcessRunner(list(self.tasks.values())).results
        
        res_dict = results[0]
        
        # Have something like:
        #     {'Exception': ValueError('ValueError raised by task_1'), 
        #      'Traceback': 'Traceback (most recent call last):\n  File "...
        #      }

        self.assertEqual(res_dict['Exception'].args, ('ValueError raised by task_1',))
        self.assertTrue(res_dict['Traceback'].startswith('Traceback (most'))
                         
        print(results)

    #------------------------------------
    # test_worker_termination
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_worker_termination(self):

        self.make_basic_task(WorkerBehavior.INFINITE_LOOP)
        
        task = list(self.tasks.values())[0]
        
        mp_runner = MultiProcessRunner(task, synchronous=False) 
        time.sleep(5)
        mp_runner.terminate_task(task)
        
        self.assertDictEqual(mp_runner.running_tasks, {})

# ------------------- Utilities -------------------

    #------------------------------------
    # make_task
    #-------------------
    
    def make_basic_task(self, behavior=WorkerBehavior.BASIC):
        
        num_existing_tasks = len(self.tasks)
        task_name = f"task_{1+num_existing_tasks}"
        
        todo_obj = BusyBody(task_name)
        
        if behavior == WorkerBehavior.BASIC:
            run_method = todo_obj.run
        elif behavior == WorkerBehavior.RAISE_EXCEPTION:
            run_method = todo_obj.run_raising_exception
        elif behavior == WorkerBehavior.INFINITE_LOOP:
            run_method = todo_obj.run_infinite_loop
        
        if behavior == WorkerBehavior.INFINITE_LOOP:
            # This one also tests no args passed to run method:
            task = Task(task_name, 
                        run_method,
                        ) 
        else:
            # Other behaviors take args and kwargs:
            task = Task(task_name, 
                        run_method,
                        f"arg1-{task_name}",  # Return arg
                        kwarg1=f"kwarg1-{task_name}"
                        ) 
        self.tasks[task_name] = task
        return task_name

# ---------------------Class BusyBody -----------

class BusyBody:

    def __init__(self, name):
        self.name = name
        signal.signal(signal.SIGTERM, self.stop_infinite_loop)
        
    def run(self, arg1, kwarg1=None):
        self.arg1 = arg1
        self.kwarg1 = kwarg1
        
        for _i in range(2):
            print(f"Job {self.name} doing stuff...")
            sleep_time = random.randint(1,2)
            time.sleep(sleep_time)
            
        return {'arg1' : self.arg1,
                'kwarg1' : self.kwarg1
                }
        
    def run_raising_exception(self, arg1, kwarg1=None):
        self.arg1 = arg1
        self.kwarg1 = kwarg1
        
        for _i in range(2):
            print(f"Job {self.name} raising value error...")
            sleep_time = random.randint(1,2)
            time.sleep(sleep_time)
            raise ValueError(f"ValueError raised by {self.name}")
            
        return {'arg1' : self.arg1,
                'kwarg1' : self.kwarg1
                }

    def stop_infinite_loop(self):
        self.keep_going = False
        
    def run_infinite_loop(self):
        
        self.keep_going = True
        while self.keep_going:
            time.sleep(1)
            print(f"Task {self.name} running infinite loop")
        print(f"Task {self.name} stopping infinite loop")
            
            

# ------------------- Main -------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()