'''
Created on Sep 6, 2021

@author: paepcke
'''

import traceback

from logging_service import LoggingService

from data_augmentation.utils import Utils
import multiprocessing as mp


# TODO:
#   o Class level doc
class MultiProcessRunner:
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, task_specs, num_workers=None, synchronous=True):
        '''
        Given a list of ready-to-go Task instances,
        run the tasks in parallel, each on its own CPU. 
        
        Use num_workers CPUs if that many are available.
        If num_workers is None, use Utils.MAX_PERC_OF_CORES_TO_USE
        percent of available cores.
        
        Tasks must return dicts. 
        
        This method returns when all tasks are done, unless
        synchronous is set to False. In that case, returns once
        all tasks have been submitted to a CPU. Submission can
        take an arbitrary amount of time, because the runner
        may need to wait for CPUs to free up.
         
        A list of dicts with each task's returned result dict
        is available in <inst>.results after this constructor
        returns from synchronous operation. 
        
        For asynch op, clients can use the running_tasks() method. 
        It returns a dict mapping Task instances (i.e. client task_specs 
        that were passed into this method) to Event instances. See
        Python 3 threading.Event. Tasks set() their respective Event 
        flag when they are done. Clients can wait() on the event instance.
        Once a task has set() its event flag, task.shared_return_dict provides 
        results of the task.

        :param task_specs: list of Task specifications
        :type task_specs: Task
        :param num_workers: max number of CPUs to use
        :type num_workers: {None | int}
        :param synchronous: if True, returns when all
            tasks are completed. Else returns when all
            tasks have been submitted
        :type synchronous: bool
        '''

        self.log = LoggingService()
        
        if type(task_specs) != list:
            task_specs = [task_specs]
            
        self.synchronous = synchronous
        
        # Determine number of workers:
        num_cores = mp.cpu_count()
        # Use only a percentage of the cores:
        if num_workers is None:
            num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            num_workers = num_cores

        # Get a list of multiprocessor dict 
        # proxy instances. They behave like dicts, but
        # turn them into basic dicts anyway. For example:
        # unittest assertDictEqual(d1, d2) complains about
        # dict not being of the proper type:
        
        results = self.run_jobs(task_specs, num_workers)
        
        if not synchronous:
            self.results = None
            return
        
        res_dicts = []
        for dict_proxy in results:
            # Make a native-Python dict from the dict proxy:
            res_dicts.append({k : v for k,v in dict_proxy.items()})
            
        # Make result list available
        self.results = res_dicts

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
        
        mp_ctx = mp.get_context('spawn')
        #mp_ctx = mp.get_context('fork')
        self.manager = mp_ctx.Manager()
        
        num_tasks  = len(task_specs)
        # Save a shallow copy of the task list, 
        # b/c we will pop() from this list
        # later:
        all_task_specs = task_specs.copy()
        
        cpu_budget = min(num_tasks, num_workers)
        self.log.info(f"Working on {len(task_specs)} task(s) using {cpu_budget} CPU(s)")

        self.cpus_available = cpu_budget
        
        # Map of Task instance to manager Event instance:
        self._running_tasks = {}
        
        # Mapping task instances to job instances;
        # used to terminate jobs by task:
        self.all_jobs = {}
        
        # Start all the workers:
        while len(task_specs) > 0:
            if self.cpus_available > 0: 
                
                task = task_specs.pop()
                
                if type(task) != Task:
                    raise TypeError(f"Tasks must be of type Task, not {type(task)} ({task})")
                
                ret_value_slot = mp.Value("b", False)
                done_event = self.manager.Event()
                shared_return_dict = self.manager.dict()
                # Make the task instance available
                # to the receiver:
                shared_return_dict['_kwargs'] = task.func_kwargs
                shared_return_dict['_done_event'] = done_event
                task.shared_return_dict = shared_return_dict
                
                job = mp_ctx.Process(target=task.run,
                                     name=task.name,
                                     #**********args=(done_event, shared_return_dict)
                                     args=(shared_return_dict,)
                                     )

                job.ret_value_slot = ret_value_slot
                self.all_jobs[task] = job
                
                self._running_tasks[task] = done_event
                job.start()
                self.cpus_available -= 1
                
            else:
                # Wait for a CPU to become available:
                _done_task = self._await_any_job_done() 

        # All tasks have been given to a CPU.
        # Wait for everyone to finish, unless client
        # instantiated this runner with synchronous set to False:
        
        if not self.synchronous:
            return
        
        self.log.info(f"Going into wait loop for {len(self._running_tasks)} still running tasks")
        while len(self._running_tasks) > 0:
            completed_task = self._await_any_job_done()
            self.log.info(f"Task {completed_task.name} finished")

        # Return a dict whose keys are task names, and vals 
        # are the return values. Note that the return 
        # values maybe be exception objects. Clients
        # need to check:

        results = [task.shared_return_dict for task in all_task_specs]
        return results

    #------------------------------------
    # terminate_task
    #-------------------
    
    def terminate_task(self, task_spec):
        '''
        Given a task spec instance, kill the
        child process that is working on this 
        task with a SIGTERM
        
        :param task_spec: task specification instance
        :type task_spec: Task
        :return True if a process was running that worked on
            the given task, and a SIGTERM was delivered. Else False
        :rtype: bool
        :raise TypeError if task_spec not instance of Task
        :raise RuntimeError if SIGTERM could not be delivered
        '''
        
        if type(task_spec) != Task:
            raise TypeError(f"Task spec must be a Task instance, not {task_spec}")
        
        # Does this job have a running 
        # process working on it?
        try:
            self._running_tasks[task_spec]
        except KeyError:
            # No child running
            return False
            
        try:
            self.log.info(f"Terminating worker {task_spec.name}")
            # Get the Process instance that is running
            # the task:
            job = self.all_jobs[task_spec]
            job.terminate()
            del self.all_jobs[task_spec]
            del self._running_tasks[task_spec]
            # Successfully delivered SIGTERM.
            # Hopefully the child processe receives
            # it:
            return True
        except Exception as e:
            # Something went wrong sending the SIGTERM
            raise RuntimeError(f"Could not send SIGTERM to {task_spec}: {repr(e)}")

    #------------------------------------
    # running_tasks
    #-------------------
    
    def running_tasks(self):
        '''
        Return dict that maps client Task instances
        to multiprocessor.manager.Event instances.
        Clients can thereby:
           o discern how many tasks are still being
               worked on
           o wait for any or all of the tasks to finish
           
        NOTE: the default sychronous operation takes
              care of such waiting.
        
        :return dict mapping Task instances to Event instances
        :rtype {Task : Event} 
        '''
        #return list(self._running_tasks.values())
        return self._running_tasks

    #------------------------------------
    # join
    #-------------------
    
    def join(self):
        '''
        Return when all tasks have finished
        Needed only by clients who use this
        facility in synchronous=False mode.
        
        :returns possibly empty list of task instances of finished tasks
        :rtype (Task) 
        '''
        
        finished_jobs = []
        
        #****************
        print("******** Enter join loop")
        #****************
        
        while len(self.running_tasks()) > 0:
            
            #****************
            print("******** Inside join loop")
            #****************
            
            finished_jobs.append(self._await_any_job_done())

        #****************
        print("******** Exit join loop")
        #****************
        
        return finished_jobs

    #------------------------------------
    # _await_any_job_done
    #-------------------
    
    def _await_any_job_done(self):
        '''
        Wait for the 'I am done' Event flag to
        be set in any of the running tasks. Return
        the first task object that is done.
        
        :return first task that has finished
        :rtype Task
        '''
        while True:
            for task_obj, done_event in self._running_tasks.items():
                # Wait a short while for this task to finish...
                task_is_done = done_event.wait(1.0) # seconds
                if task_is_done:
                    del self._running_tasks[task_obj]
                    self.cpus_available += 1
                    return task_obj

# --------------- Task Class --------------------

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
                 *args, 
                 **kwargs):

        self.name = name
        self.target_func = target_func
        self.func_args = args
        # The func_args and func_kwargs will be passed to the
        # target func:
        self.func_kwargs = kwargs
        for kwarg_name, kwarg_val in kwargs.items():
            self.__setattr__(kwarg_name, kwarg_val)

    #------------------------------------
    # run
    #-------------------
    
    def run(self, shared_return_dict):

        # Last-minute additions to the kwargs of
        # the target function would be in 
        # shared_return_dict['_kwargs']. Merge those
        # into the self.func_kwargs dict that will be
        # passed to the target function:
        
        self.func_kwargs.update(shared_return_dict['_kwargs'])

        try:
            res = self.target_func(*self.func_args, **self.func_kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            res = {'Exception' : e,
                   'Traceback' : tb
                   }
            
        if type(res) == dict:
            shared_return_dict.update(res)
        else:
            #************
            print(f"******** res: {res}")
            #************
            shared_return_dict['result'] = res
        # Indicate to the outside world that
        # we are done:
        
        #************
        print(f"******** setting flag")
        #************
        
        shared_return_dict['_done_event'].set()

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