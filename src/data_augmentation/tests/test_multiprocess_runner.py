'''
Created on Sep 6, 2021

@author: paepcke
'''
import random
import time
import unittest

from data_augmentation.multiprocess_runner import MultiProcessRunner, Task, ReturnType

TEST_ALL = True
#TEST_ALL = False

class MultiProcessRunnerTester(unittest.TestCase):


    def setUp(self):
        self.tasks = {}


    def tearDown(self):
        pass

# ------------------ Tests -----------------

    #------------------------------------
    # test_basics
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_basics(self):
        
        task_name = self.make_basic_task()
        results = MultiProcessRunner(self.tasks[task_name])
        print(results)

# ------------------- Utilities -------------------

    #------------------------------------
    # make_task
    #-------------------
    
    def make_basic_task(self):
        
        num_existing_tasks = len(self.tasks)
        task_name = f"task_{1+num_existing_tasks}"
        
        todo_obj = BusyBody(task_name)
        task = Task(task_name, 
                    todo_obj.run,
                    ReturnType.VALUE,
                    f"arg1-{task_name}",  # Return arg
                    kwarg1=f"kwarg1-{task_name}"
                    ) 
        self.tasks[task_name] = task
        return task_name

# ---------------------Class BusyBody -----------

class BusyBody:
    
    def __init__(self, name):
        self.name = name
        
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
        
        

# ------------------- Main -------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()