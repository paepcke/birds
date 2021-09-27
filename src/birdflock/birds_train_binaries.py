#!/usr/bin/env python3
'''
Created on Sep 9, 2021

@author: paepcke
'''
import datetime
import os
from pathlib import Path
import random

import pandas as pd

from experiment_manager.experiment_manager import ExperimentManager, Datatype
from experiment_manager.neural_net_config import NeuralNetConfig, ConfigError
from logging_service import LoggingService
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
                 timestamp=None
                 ):
        '''
        Train a flock of binary bird call classifiers in parallel,
        optionally balancing each focal species against the other
        samples.
        
        Which species are in turn taken as a focal species for
        training a binary focalSpecies-against-all is controlled as
        follows:
        
           o The config file entry 'root_train_test_data' in section
             'Paths' is taken as the root of the training snippets.
             The snippets must be in subdirectories below that path,
             one subdir per species
           o focals_list optionally limits for which species a 
             classifier is trained. The 'others' may still 
             include all of species_list or the subdirectories of 
             snippets_root.
    
        :param config_info: a path to the files with all path 
            and training parameters, or a config instance if
            client has already loaded from a config file
        :type config_info: {str | NeuralNetConfig}
        :param focals_list: optional list of species to train for;
            if None, classifiers for all species are created
        :type focals_list: {None | [str]}
        :param experiment_path: if provided the directory in which
            ExperimentManager experiments are created. Default
            is <this-script's-directory>/Experiments
        :type experiment_path: {None | str}
        :param timestamp: optionally a datetime string to use
            when creating ExperimentManager experiment file names
        :type timestamp {None | str}
        '''

        self.log = LoggingService()
        
        try:
            self.config = self._initialize_config_struct(config_info)
        except Exception as e:
            msg = f"During config init: {repr(e)}"
            self.log.err(msg)
            raise RuntimeError(msg) from e

        # Check timestamp for proper format:
        if timestamp is not None:
            test_ts = Utils.timestamp_from_exp_path(timestamp)
            if test_ts != timestamp:
                raise ValueError(f"Timestamp must be of the form 2021-09-20T10_16_59, not {timestamp}")

        self.snippets_root = self.config['Paths']['root_train_test_data']
        self.set_seed(self.config.Training.getint('seed', 42))

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

        self.focals_list = focals_list
        
        if experiment_path is None:
            experiment_path = os.path.join(os.path.dirname(__file__), 'Experiments')
        self.experiment_path = experiment_path
        
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
            # Grab all GPUs:
            self.gpu_pool = list(range(num_gpus))

        self.tasks_to_run = self._create_task_list(focals_list, timestamp)

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
                tasks_left -= set([task])
                

                if len(self.gpu_pool) > 0 and \
                   len(tasks_left) > 0:
                    # Add another task to the task batch:
                    continue
                
                # Task batch is ready.
                # Fire up the parallel training sessions,
                # as many as we have GPUs. Don't wait for
                # them to finish, because if some finish
                # early, we want to train the next:
                self.log.info((f"Start training classifier(s) "
                               f"{[task.name for task in task_batch]}" 
                               f"on GPUs {[task.gpu for task in task_batch]}")
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
    
    def _create_task_list(self, focals_list, timestamp=None):
        '''
        Create a list of experiment_manager.Task instances
        to run trainings in parallel. If timestamp is provided,
        it must be precisely of the ISO form:

            2021-09-24T10_42_55
        
        This datetime string will be used when generating
        result files such as:

             Classifier_GRASG_2021-09-24T10_42_55
             
        and later corresponding inference experiments;
        
             Classifier_GRASG_2021-09-24T10_42_55_inference
             
        If None, a current timestamp is created.
        
        :param focals_list: list of species for which to 
            train binary classifiers.
        :type focals_list: [str]
        :param timestamp: datetime to use for all experiment
            file names
        :type timestamp: {None | str}
        '''
    
        task_list = []
        # Timestamp to use in all the experiment directory
        # names: 
        #    .../Experiment/Classifier_<species>_<common_time_stamp>
        if timestamp is None:
            common_time_stamp = FileUtils.file_timestamp()
        else:
            common_time_stamp = timestamp

        gpu_list = self.gpu_pool.copy()
        
        for species in focals_list:
            
            try:
                # Use GPUs round-robin:
                gpu_this_task = gpu_list.pop()
            except IndexError:
                gpu_list = self.gpu_pool.copy()
                gpu_this_task = gpu_list.pop()
                
            task = Task(species,                  # Name of task
                        self._create_classifier,  # function to execute
                        self.config,              # overall configuration (arg to task)
                        species,                  # name of species (arg to task)
                        common_time_stamp,        # part of experiment directory filename (arg to task)
                        gpu_this_task             # GPU this task will use (arg to task)
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
                          config, 
                          target_species,
                          common_time_stamp,
                          gpu=None):

        # Create a dataset with target_species
        # against everyone else, and train it:

        experiments_root = os.path.join(self.experiment_path,
                                        f"Classifier_{target_species}_{common_time_stamp}")
        experiment = ExperimentManager(experiments_root)
        
        # Remember the network configuration:
        experiment.save('hparams', self.config)
        experiment['class_label_names'] = [target_species]
        experiment.save()

        clf = BinaryClassificationTrainer(config,
                                          target_species,
                                          device=gpu,
                                          experiment=experiment
                                          )
        
        net = clf.net
        experiment.save(target_species, net)
        self._save_classifier_history(net.history, experiment)
        
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
    # _save_classifier_history
    #-------------------
    
    def _save_classifier_history(self, history, experiment):
        '''
        Given the history of a skorch classifier, remove some
        items, then save as csv in the given experiment under
        the name <species>_res_by_epoch.csv
        
        The history has the following form:

            [{'batches':
                 [{'train_loss': 0.7793650031089783,
                   'train_batch_size': 6},
                  {'valid_loss': 74.49720001220703,
                   'valid_batch_size': 1}
                   ],
              'epoch': 1,
              'train_batch_count': 1,
              'valid_batch_count': 1,
              'dur': 1.7713217735290527,
              'train_loss': 0.7793650031089783,
              'train_loss_best': True,
              'valid_loss': 74.49720001220703,
              'valid_loss_best': True,
              'valid_acc': 0.0,
              'valid_acc_best': True,
              'balanced_accuracy': 0.0,
              'balanced_accuracy_best': True,
              'f1': 0.0,
              'f1_best': True,
              'accuracy': 0.0,
              'accuracy_best': True,
              'species': 'YOFLG'
              },
            
              {'batches': [{'train_loss': 0.5075358748435974, 'train_batch_size': 6},
                           {'valid_loss': 0.0, 'valid_batch_size': 1}],
               'epoch': 2,
               'train_batch_count': 1, 'valid_batch_count': ...
            ]

        We remove the batches information, and move the species
        to the front, creating a new dict for saving.
        
        The result is a df like:
        
		      species  train_loss train_loss_best  ...  f1_best accuracy  ...
		Epoch
		1       VASEG    0.752131            True  ...     True      0.0  
		2       VASEG    0.058150            True  ...    False      0.0  
		3       VASEG    0.220424           False  ...    False      1.0  
		4       VASEG    0.933877           False  ...    False      1.0  
		5       VASEG    0.679428           False  ...    False      1.0  
		6       VASEG    0.375119           False  ...    False      1.0  
				
        For this example, the info is stored in the given experiments
        tabular info under key VASEG_res_by_epoch, and can be
        retrieved as a df as:
        
            experiment.read('VASEB_res_by_epoch', Datatype.tabular)

        :param history: history of skorch results during training
        :type history: [{str : Any}]
        :param experiment: experiment manager to which to save
        :type experiment: ExperimentManager
        '''

        # Get the col names of the history,
        # such as 'train_loss', 'train_loss_best', ...
        # Use the keys of the first history entry
        # (all entries will have the same keys
        # and species:
        first_hist_entry = history[0]
        species = first_hist_entry['species']

        hist_keys = list(first_hist_entry.keys())
        
        # Remove columns we don't care about: 
        hist_keys.remove('batches')
        hist_keys.remove('dur')
        hist_keys.remove('train_batch_count')
        hist_keys.remove('valid_batch_count')
        
        # History has the species at the end;
        # we'll move it to be the first col in 
        # the results:
        hist_keys.remove('species')
        # Epoch will be the auto-generated index
        # of 0,1,2,...
        hist_keys.remove('epoch')
        
        # Columns in the result df, and 
        # also the keys with which we will pick
        # the results from the history:
        cols = ['species'] + hist_keys
        
        res_df = pd.DataFrame([], columns=cols)
        
        for epoch_res in history:
            epoch_res_row_dict = {}
            # Pull values in the order of the
            # columns we want:
            for key in cols:
                epoch_res_row_dict[key] = epoch_res[key]
            # Turn into a pandas Series, and append to the
            # result dataframe
            row_ser = pd.Series(epoch_res_row_dict, name=epoch_res['epoch'])
            res_df  = res_df.append(row_ser)
        
        # Write to the experiment:
        res_df.index.name = 'Epoch'
        experiment.save(f"{species}_res_by_epoch", res_df)
        return res_df

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
# Use run_training.py for training, rather than
# this commented code.

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
#                                      formatter_class=argparse.RawTextHelpFormatter,
#                                      description="Train multiple binary species classifiers."
#                                      )
#
#     parser.add_argument('-s', '--species',
#                         type=str,
#                         nargs='+',
#                         help='Repeatable: species for which to train classifiers; default: all',
#                         default=None
#                         )
#
#     parser.add_argument('species_root',
#                         help='root of spectrogram snippet subdirectories'
#                         )
#
#
#     args = parser.parse_args()
#
#
#     if not os.path.exists(args.species_root):
#         print(f"Cannot find {args.species_root}")
#         sys.exit(1)
#
#     if args.species is not None:
#         # Ensure a subdir for each species:
#         for species in args.species:
#             dir_name = os.path.join(args.species_root, species)
#             if not os.path.exists(dir_name):
#                 print(f"Cannot find {dir_name}, yet classifier for species '{species}' was requested")
#                 sys.exit(1)
#
#     BinaryBirdsTrainer(args.species_root,
#                        focals_list=args.species
#                        )
        