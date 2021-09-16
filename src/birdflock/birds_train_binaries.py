#!/usr/bin/env python3
'''
Created on Sep 9, 2021

@author: paepcke
'''
import argparse
import datetime
import os
from pathlib import Path
import random
import sys

from experiment_manager.experiment_manager import ExperimentManager
from logging_service import LoggingService
from experiment_manager.neural_net_config import NeuralNetConfig, ConfigError
from torch import cuda
import torch

from birdflock.train_one_binary_classifier import BinaryClassificationTrainer
from birdsong.utils.utilities import FileUtils
from data_augmentation.multiprocess_runner import Task, MultiProcessRunner
from data_augmentation.utils import Utils
import torch.multiprocessing as mp


class BinaryBirdsTrainer(object):
    '''
    Given the root of subdirectories with training
    samples:
       o Make each species a focal species in turn
       o For each case, create a trainer that
         trains a binary classifier focalSpecies-against-all-others.
       o Resulting models will be in the models path of
         the ExperimentManager experiment whose directory
         root is provided, or is under 'Experiment_<date-time>'
         in this file's directory
       o Init args enable requests for balancing each
         dataset to a specific ratio:
         
             num-focal-species-samples
             -------------------------
                num-other-samples
                
       o Each training runs on its own CPU and GPU
       
    After training, task instances are available in 
    attribute 'tasks_to_run'. Each task instance in that
    list will have an attribute 'error'. If it is None,
    all ran fine, else it will contain an exception instance.
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 config_info, 
                 focals_list=None,
                 experiment_path=None,
                 ):
        '''
        Train a flock of binary bird call classifiers in parallel,
        optionally balancing each focal species against the other
        samples.
        
        Which species are in turn taken as a focal species for
        training a binary focalSpecies-against-all is controlled as
        follows:
        
           o snippets_root is the required root of subdirectories, each
             of which contains spectrogram samples of one species
           o focals_list is another optional subset that limits for which 
             species a classifier is trained. The 'others' may still 
             include all of species_list or the subdirectories of snippets_root.
    
        :param config_info: a path to the files with all path 
            and training parameters, or a config instance if
            client has already loaded from a config file
        :type config_info: {str | NeuralNetConfig}
        :param focals_list:
        :type focals_list:
        :param experiment_path:
        :type experiment_path:
        '''

        self.log = LoggingService()
        
        try:
            self.config = self._initialize_config_struct(config_info)
        except Exception as e:
            msg = f"During config init: {repr(e)}"
            self.log.err(msg)
            raise RuntimeError(msg) from e

        self.snippets_root = self.config['Paths']['root_train_test_data']
        self.set_seed(self.config.Training.getint('seed', 42))

        self.focals_list = focals_list
        
        if experiment_path is None:
            experiment_path = os.path.join(os.path.dirname(__file__), 'Experiments')
        self.experiment_path = experiment_path

        # If no species_list is specified, use
        # all species subdirectories:
        if focals_list is None:
            # Collect names of species (i.e. subdirectory names),
            # i.e. species for which to create classifiers:
            focals_list = [Path(species_dir).stem
                            for species_dir
                            in Utils.listdir_abs(self.snippets_root)
                            if os.path.isdir(species_dir)
                            ]

        # GPUs not used by others as we start:
        self.gpu_pool    = []
        num_gpus    = cuda.device_count()

        if num_gpus == 0:
            # No GPUs on this machine, use
            # the CPUs instead:
            num_cpus = mp.cpu_count()
            # Leave 2 CPUs unused:
            self.gpu_pool = ['cpu']*min(len(focals_list), (num_cpus - 2))
        else:
            # Grab all unused GPUs:
            for i in range(num_gpus):
                # Check whether any process is using
                # this GPU now, before we start:
                if not self._gpu_in_use(i):
                    self.gpu_pool.append(i)

        self.tasks_to_run = self._create_task_list(focals_list)

    #------------------------------------
    # train
    #-------------------
    
    def train(self):
        
        mp_runners = []
        task_batch = []
        tasks_left = set(self.tasks_to_run.copy())
        while len(tasks_left) > 0:
            for task in self.tasks_to_run:
                gpu = self.gpu_pool.pop()
                
                # Usually gpu will be an integer GPU
                # device index. But it may be 'cpu'
                # during debugging:
                task.gpu = gpu
                
                # Add gpu to use to the kwargs that
                # will be passed to the target function
                # down the road. This value is added
                # to whatever we specified as kwargs
                # when we created the Task instance in 
                # _create_task_list()
                
                try:
                    task.func_kwargs['gpu'] = gpu
                except (KeyError, AttributeError):
                    task.func_kwargs = {'gpu' : gpu}
                
                task_batch.append(task)
                if len(self.gpu_pool) > 0:
                    # Add another task to the task batch:
                    continue
                
                # Task batch is ready.
                # Fire up the parallel training sessions,
                # as many as we have GPUs. Don't wait for
                # them to finish, because if some finish
                # early, we want to train the next:
                self.log.info((f"Start training classifier(s) "
                               f"{[task.name for task in task_batch]}")
                               )
                
                # Copy the task_batch to be sure not 
                # to interfere when we set task_batch to
                # empty further down. Use the runner with
                # synchronous=False so that it returns immediately,
                # so we can start another one or more tasks
                # with another mp_runner. We wait for them all
                # at the end.
                mp_runners.append(MultiProcessRunner(task_batch.copy(), 
                                                     synchronous=False)
                )
    
                tasks_left -= set(task_batch)
                
                # Wait for any of the classifiers to finish
                # training and free up a GPU, then run more: 
                # done_task_objs will be a set on a task when
                # it's done:
                try:
                    done_task_objs = self._await_any_job_done(task_batch)
                except Exception as e:
                    
                    # Give up on this batch of tasks. 
                    
                    # Print the msg, which will include a traceback,
                    # and use the species attribute added down
                    # in _check_task_exception for pinpointing.
                    # then continue chugging:
                    try:
                        species = e.species
                    except AttributeError:
                        species = "unavailable"
                    self.log.err(f"In task (species {species}): {repr(e)}")
                    for task in task_batch:
                        try:
                            task.error = e
                            mp_runners[-1].terminate_task(task)
                        except Exception:
                            # Maybe we can't terminate b/c already 
                            # dead...best effort:
                            pass
                        self.gpu_pool.append(task.gpu)
                    task_batch = []
                    done_task_objs = task_batch
                    done_task_names = [task.species for task in done_task_objs]
                    self.log.err(f"Continuing to next species; don't trust classifiers for {done_task_names}")
                    continue

                done_task_names = [task.species for task in done_task_objs]
                self.log.info(f"Finished classifier(s) {done_task_names}")
                
                # Return GPU to the pool:
                self.gpu_pool.extend([task.gpu for task in done_task_objs])
    
                task_batch = []
                # loop, submitting next training job(s) to 
                # a new mp_runner
                
        # Wait for all tasks in all runners to be done:
        for mp_runner in mp_runners:
            mp_runner.join()
            # Check the tasks for errors, when
            # no 'error' attribute was assigned
            # in the above loop, set 'error' to None
            # for each task:
            for task in mp_runner.task_reference().keys():
                try:
                    task.error
                except AttributeError:
                    # No error was set in the loop above; great!
                    task.error = None
            

    #------------------------------------
    # tasks
    #-------------------

    @property
    def tasks(self):
        return [task.species for task in self.tasks_to_run]

    #------------------------------------
    # set_seed  
    #-------------------

    def set_seed(self, seed):
        '''
        Set the seed across all different necessary platforms
        to allow for comparison of different models and runs
        
        :param seed: random seed to set for all random num generators
        :type seed: int
        '''
        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)

    #------------------------------------
    # _create_task_list
    #-------------------
    
    def _create_task_list(self, focals_list):
    
        task_list = []
        for species in focals_list:
            task = Task(species,                  # Name of task
                        self._create_classifier,  # function to execute
                        self.config,              # where the snippets are
                        species                   # name of species as arg to task
                        )
            # Species at play to know by the task itself.
            # The species are in the Task() instantiation
            # above will be for the _create_classifier() call
            # at runtime:
            task.species = species
            task_list.append(task)
        return task_list

    #------------------------------------
    # _await_any_job_done
    #-------------------
    
    def _await_any_job_done(self, task_obj_list, extra_time=None): 
        '''
        Given a list of task objects, monitor their
        multiprocessing.Event instances. When one
        or more tasks are done as evidenced by them
        having set the event flag, return the list of
        done tasks.
        
        The extra_time determines how much extra time should
        be allowed for tasks to finish after detecting one
        finished task. If none, returns with just one task
        done. 
        
        returns list of task(s) objects that are done
        training.
        
        :param task_obj_list: list of tasks that are to
            be monitored
        :type task_obj_list: (Task)
        :param extra_time: seconds to wait beyond detecting
            a task having finished
        :type extra_time: {None | seconds}
        :returns list of done tasks
        :rtype: (Task)
        '''
        start_time = 0
        # Keep checking task event flags round
        # robin till one is done:
        tasks_done = set()
        task_obj_set = set(task_obj_list)
        while True:
            for task_obj in task_obj_list:
                
                if extra_time is not None \
                    and len(tasks_done) > 0 \
                    and (datetime.datetime.now() - start_time).seconds >= extra_time:
                    return tasks_done
                
                # Already saw that this task is done?
                if task_obj in tasks_done:
                    # Examine next task:
                    continue
                if task_obj.shared_return_dict['_done_event'].wait(3.0): # seconds to wait
                    # Task was finished:
                    tasks_done.add(task_obj)
                    # If task signaled an error, raise
                    # it now, else it will be hidden away:
                    self._check_task_exception(task_obj)
                    
                    if extra_time is None:
                        # Return the first done task right away
                        return tasks_done
                    else:
                        start_time = datetime.datetime.now()
                        continue
                else:
                    # Task was not done: examine next task:
                    continue # for...
            
            # Checked all in task_obj_list;
            # sweep them again if any left.
            # Update the task_obj_list to remove
            # the tasks found to be done:
            task_obj_set = set(task_obj_set) - set(tasks_done)
            
            if len(task_obj_set) > 0:
                continue
            return tasks_done

    #------------------------------------
    # _check_task_exception
    #-------------------
    
    def _check_task_exception(self, task):
        task_keys = task.shared_return_dict.keys()
        if 'Exception' in task_keys and 'Traceback' in task_keys:
            msg = (f"Exception in trainer for {task.name}:device_{task.gpu}\n",
                   task.shared_return_dict['Traceback']
                   )
            e = RuntimeError(msg)
            # Add attribute 'species'
            # for the higher-ups to add info: 
            e.species = task.species
            raise e

    #------------------------------------
    # _create_classifier
    #-------------------
    
    def _create_classifier(self, 
                          snippets_root, 
                          target_species,
                          balancing_strategy=None,
                          balancing_ratio=None,
                          gpu=None):

        # Create a dataset with target_species
        # against everyone else, and train it:

        experiments_root = os.path.join(self.experiment_path,
                                        f"Classifier_{target_species}_{FileUtils.file_timestamp()}")
        experiment = ExperimentManager(experiments_root)

        clf = BinaryClassificationTrainer(snippets_root,
                                          target_species,
                                          device=gpu,
                                          experiment=experiment
                                          )
        
        net = clf.net
        experiment.save(target_species, net)
        
        # Note some settings:*******
        
        # Find the task instance we just finished,
        # and signal its completion:
        for task in self.tasks_to_run:
            if task.name == target_species:
                break

        # NOTE: cannot return clf itself,
        #       because it cannot be pickled.
        task.shared_return_dict['result'] = task.species
        task.shared_return_dict['_done_event'].set()

# ------------------- Utilities -----------------

    #------------------------------------
    # 
    #-------------------


    #------------------------------------
    # _gpu_in_use
    #-------------------
    
    def _gpu_in_use(self, device_num):
        procs = torch.cuda.list_gpu_processes(device_num)
        if procs.find('no processes are running') > -1:
            return False
        else:
            return True

    #------------------------------------
    # _initialize_config_struct 
    #-------------------
    
    def _initialize_config_struct(self, config_info):
        '''
        Initialize a config dict of dict with
        the application's configurations. Sections
        will be:
        
          config['Paths']       -> dict[attr : val]
          config['Training']    -> dict[attr : val]
          config['Parallelism'] -> dict[attr : val]
        
        The config read method will handle config_info
        being None. 
        
        If config_info is a string, it is assumed either 
        to be a file containing the configuration, or
        a JSON string that defines the config.
         
        Else config_info is assumed to be a NeuralNetConfig.
        The latter is relevant only if using this file
        as a library, rather than a command line tool.
        
        If given a NeuralNetConfig instance, it is returned
        unchanged. 
        
        :param config_info: the information needed to construct
            the structure
        :type config_info: {NeuralNetConfig | str}
        :return a NeuralNetConfig instance with all parms
            initialized
        :rtype NeuralNetConfig
        '''

        if isinstance(config_info, str):
            # Is it a JSON str? Should have a better test!
            if config_info.startswith('{'):
                # JSON String:
                config = NeuralNetConfig.from_json(config_info)
            else: 
                config = Utils.read_configuration(config_info)
        elif isinstance(config_info, NeuralNetConfig):
            config = config_info
        else:
            msg = f"Error: must have a config file, not {config_info}. See config.cfg.Example in project root"
            # Since logdir may be in config, need to use print here:
            print(msg)
            raise ConfigError(msg)
            
        return config


# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Train multiple binary species classifiers."
                                     )

    parser.add_argument('-s', '--species',
                        type=str,
                        nargs='+',
                        help='Repeatable: species for which to train classifiers; default: all',
                        default=None
                        )

    parser.add_argument('species_root',
                        help='root of spectrogram snippet subdirectories'
                        )


    args = parser.parse_args()


    if not os.path.exists(args.species_root):
        print(f"Cannot find {args.species_root}")
        sys.exit(1)
        
    if args.species is not None:
        # Ensure a subdir for each species:
        for species in args.species:
            dir_name = os.path.join(args.species_root, species)
            if not os.path.exists(dir_name):
                print(f"Cannot find {dir_name}, yet classifier for species '{species}' was requested")
                sys.exit(1)
                
    BinaryBirdsTrainer(args.species_root,
                       species_list=args.species
                       )
        