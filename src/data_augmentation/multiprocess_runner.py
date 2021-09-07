'''
Created on Sep 6, 2021

@author: paepcke
'''

from logging_service import LoggingService

from data_augmentation.utils import Utils
import multiprocessing as mp

from enum import Enum


class MultiProcessRunner:
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, task_specs, num_workers=None):

        self.log = LoggingService()
        
        if type(task_specs) != list:
            task_specs = [task_specs]
        
        # Determine number of workers:
        num_cores = mp.cpu_count()
        # Use only a percentage of the cores:
        if num_workers is None:
            num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            num_workers = num_cores

        self.run_jobs(task_specs, num_workers)
        
        
    #------------------------------------
    # run_jobs
    #-------------------

    def run_jobs(self, task_specs, num_workers):
        '''
        Create processes on multiple CPUs, and feed
        them augmentation tasks. Wait for them to finish.
        
        :param task_specs: list of AugmentationTask instances
        :type task_specs: [AugmentationTask]
        :param num_workers: number of CPUs to use simultaneously.
            Default is 
        :type num_workers: {None | int}
        '''
        
        if len(task_specs) == 0:
            self.log.warn("Audio augmentation task list was empty; nothing done.")
            return
        
        # For nice progress reports, create a shared
        # dict quantities that each process will report
        # on to the console as progress indications:
        #    o Which job-completions have already been reported
        #      to the console (a list)
        # All Python processes on the various cores
        # will have read/write access:
        manager = mp.Manager()
        
        num_tasks  = len(task_specs)
        
        cpu_budget = min(num_tasks, num_workers)
        self.log.info(f"Working on {len(task_specs)} task(s) using {cpu_budget} CPU(s)")

        cpus_available = cpu_budget
        running_tasks = {}
        all_jobs = []
        
        # Start all the workers:
        while len(task_specs) > 0:
            if cpus_available > 0: 
                
                task = task_specs.pop()
                ret_value_slot = mp.Value("b", False)
                done_event = manager.Event()
                shared_return_struct = self.get_return_structure(manager, task)
                job = mp.Process(target=task.run,
                                 args=(done_event, shared_return_struct)
                                 )

                job.ret_value_slot = ret_value_slot
                all_jobs.append(job)
                
                running_tasks[task] = done_event
                job.start()
                cpus_available -= 1
                
            else:
                # Wait for a CPU to become available:
                _done_task = self.await_any_job_done(running_tasks) 

        # All tasks have been given to a CPU.
        # Wait for everyone to finish:
        self.log.info(f"Going into wait loop for {len(running_tasks)} still running tasks")
        while len(running_tasks) > 0:
            completed_task = self.await_any_job_done(running_tasks)
            self.log.info(f"Task {completed_task.name} finished")
            del running_tasks[completed_task]

        # Return a dict whose keys are task names, and vals 
        # are the return values. Note that the return 
        # values maybe be exception objects. Clients
        # need to check:
        results = [job.ret_value_slot for job in all_jobs]
        return results

    #------------------------------------
    # await_any_job_done
    #-------------------
    
    def await_any_job_done(self, running_tasks):
        '''
    
        :param running_tasks:
        :type running_tasks:
        '''
        while True:
            for task_obj, done_event in running_tasks.items():
                # Wait a short while for this task to finish...
                task_is_done = done_event.wait(1.0) # seconds
                if task_is_done:
                    return task_obj

    #------------------------------------
    # get_return_structure
    #-------------------
    
    def get_return_structure(self, mp_manager, task):
        
        struct = task.return_type
        
        if type(struct) != ReturnType:
            raise TypeError(f"Task return types must of a ReturnType enum, not {struct}")
        
        if struct == ReturnType.LIST:
            return mp_manager.list()
        elif struct == ReturnType.DICT:
            return mp_manager.dict()
        elif struct == ReturnType.VALUE:
            return mp_manager.Value()
        elif struct == ReturnType.ARRAY:
            return mp_manager.Array()
        elif struct == ReturnType.QUEUE:
            return mp_manager.Queue()

# --------------- Task Class --------------------

class ReturnType(Enum):
    LIST = 1
    DICT = 2
    VALUE = 3
    ARRAY = 4
    QUEUE = 5

class Task(dict):
    '''
    All information needed to run one task
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 name, 
                 target_func,
                 return_type, 
                 *args, **kwargs):

        self.name = name
        self.target_func = target_func
        self.func_args = args
        self.func_kwargs = kwargs
        self.return_type = return_type

    #------------------------------------
    # run
    #-------------------
    
    def run(self, done_event, shared_return_struct):

        # Though done_event and shared_return_struct are
        # used only in this method, make them inspectable
        # from the outside:
        self.done_event = done_event
        self.shared_return_struct = shared_return_struct
        res = self.target_func(self.func_args, self.func_kwargs)
        self.shared_return_struct = res
        self.done_event.set()

    #------------------------------------
    # done
    #-------------------

    def done(self):
        try:
            return self.ret_value_slot
        except KeyError:
            raise RuntimeError(f"Called done() on task '{self.name}' which has not been started yet.")

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        return f"<Task {self.name} {hex(id(self))}>"

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()

    #------------------------------------
    # __hash__
    #-------------------
    
    def __hash__(self):
        return hash(repr(self))

# -------------------------- Main -------------

if __name__ == '__main__':
    pass