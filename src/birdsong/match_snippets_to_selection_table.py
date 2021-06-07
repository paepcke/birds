#!/usr/bin/env python3
'''
Created on May 13, 2021

@author: paepcke
'''
from _collections_abc import Iterable
import argparse
from bisect import bisect_right
import datetime
import inspect
from logging import INFO
import os
from pathlib import Path
import re
import sys
import types

from logging_service.logging_service import LoggingService

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import multiprocessing as mp


# TODO
#    - Snippets are loaded twice, which is slow!
#      Once in method snips_iterator() of class 
#      SelTblSnipsAssoc. Then again in method match_snippet()
#      of class SnippetSelectionTableMapper. The first
#      time the load is to get each snippet's metadata so
#      that the snippets can be fed out by start time.
#      The second time is to copy the img to an outdir,
#      with augmented metadata (species added).
class SnippetSelectionTableMapper:
    '''
    
    Matches and labels spectrogram snippets with 
    entries in a Raven selection table. 'Matching'
    means a time overlap between the snippet's begin
    and end times, and a selection table row's begin
    and end time.
    
    The end goal is for each available selection table
    to 'find' and augment the metadata of the snippets 
    from the field recording that the table covers. Each 
    snippet's embedded metadata is enriched with information 
    from the table, such as frequency, species, and more.
    
    In the overall workflow, this module fits as follows:
    
         o Field recording R is taped in the field
         o R is loaded into Raven, or other sound labeling
              software. A Selection Table is manually created
              and exported. Each row contains start and end
              times of an observation, plus frequencies, etc.
         o Table and a spectrogram S_R of recording R are exported
         o S_R is chopped into 6-sec 'snippet' images
         
         o Using this SnippetSelectionTableMapper class,
           each snippet's time interval is matched to the
           time intervals in the selection table rows. When
           a snippet's interval overlaps a row's observation interval,
           a match has occurred. If no row is found that matches
           a snippet, the snippet is assumed to contain environment
           noise.
           
           A matched snippet's embedded metadata is augmented
           with information from the matching row (see below).
    
    Raven selection tables for birds are csv files:

        Selection           row number
        View                not used
        Channel             not used
        Begin Time (s)      begin of vocalization in fractional seconds
        End Time (s)        end of vocalization in fractional seconds
        Low Freq (Hz)       lowest frequency within the lassoed vocalization
        High Freq (Hz)      highest frequency within the lassoed vocalization
        species             four-letter species name
        type                {song, call, call-1, call-trill}
        number              not used
        mix                 comma separated list of other audible species
        
    A non-empty mix column indicates that multiple observations
    are detected at the same time in the recording. In this case,
    a copy of a matching snippet is made for each element in the
    mix list. Each snippet replica's 'species' metadata field is
    set to one of the mix members.

    At the outset, snippet metadata is expected to look like this:
    
		 sr : 22050
		 duration : 115.0
		 species : batpig1
		 duration(secs) : 4.990916431166734
		 start_time(secs) : 61.8873637464675
		 end_time(secs) : 66.87828017763424
    

    Snippet metadata added in case of a match:
    
         'low_freq'          : <low bound of frequency involved in vocalization
         'high_freq'         : <high bound of frequency involved in vocalization
         'multiple_species'  : <comma-separated list of species heard simultaneously>
         'type'              : <whether Song/Call/Call-1/Call-Trill...>

    IMPORTANT: procedures rely on each spectrogram
    	snippet's metadata to include the time range
    	that the snippet covers in the larger spectrogram
    	from which it is snipped. This metadata is added
    	to the .png files created by chop_spectrograms.py

    '''

    TASK_QUEUE_SIZE=1000
    '''Number of snippet matching tasks to submit before waiting for some to be finished'''

    RECORDING_ID_PAT = re.compile(r'[_]{0,1}(AM[0-9]{2}_[0-9]{8}_[0-9]{6})')
    '''Regex pattern to identify audio moth recording identifiers embedded in filenames.
       Ex.: DS_AM01_20190719_063242.png
            /foo/bar/AM01_20190719_063242_start.txt
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 selection_tbl_loc, 
                 spectrogram_locs, 
                 outdir,
                 global_info,
                 num_workers=None,
                 unittesting=False
                 ):
        '''
        Create snippet copies into outdir 
        for all snippets that are covered
        by any of the given selection tables.
        
        :param selection_tbl_loc: path to individual selection
            table or a directory containing selection tables.
            Each tbl is a tsv file with extension .txt
        :type selection_tbl_loc: str
        :param spectrogram_locs: individual or directory of 
            spectrogram snippets.
        :type spectrogram_locs: str
        :param outdir: destination of snippet copies
        :type outdir: src
        :param unittesting: if True, does not initialize
            the instance, or run any operations
        :type unittesting: bool
        '''

        if unittesting:
            return
        
        if not os.path.exists(selection_tbl_loc):
            print(f"Cannot open {selection_tbl_loc}")
            sys.exit(1)
            
        sel_tbl_paths = self.make_sel_tbl_dict(selection_tbl_loc)
        if len(sel_tbl_paths) == 0:
            print (f"No selection table(s) found at {selection_tbl_loc}")
            sys.exit(1)
        
        if not os.path.exists(spectrogram_locs):
            print(f"Spectrogram snippets {spectrogram_locs} not found")
            sys.exit(1)

        self.log = LoggingService()

        # Is snippets path to an individual .png snippet file?
        if os.path.isfile(spectrogram_locs):
            snippet_paths = iter([spectrogram_locs])
        else:
            # Caller gave directory of .png files.
            # Get them all recursively:
            snippet_paths = Utils.find_in_tree_gen(spectrogram_locs, 
                                                    pattern="*.png")
        # If outdir does not exist, create it,
        # and all dirs along the path:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Get dict:
        #
        #    {<recording-id> : SelTblSnipsAssoc-instance}
        #
        # where each SelTblSnipsAssoc instance is a generator
        # of snippet metadata from snippets that are covered in 
        # the selection table that is associated with the instance.
        # In addition the absolute snippet path is added to that
        # metadata as entry with key 'snip_path'.
        # 
        # The generator feeds out the snippet metadata in order of
        # start time.
        #
        # For brevity, call each instance of SelTblSnipsAssoc
        # an 'assoc'

        rec_id_assocs = self.create_snips_gen_for_sel_tbls(
            snippet_paths, 
            iter(list(sel_tbl_paths.values())))
        
        num_cores = mp.cpu_count()
        # Use only a percentage of the cores:
        if num_workers is None:
            num_workers = round(num_cores * Utils.MAX_PERC_OF_CORES_TO_USE  / 100)
        elif num_workers > num_cores:
            # Limit pool size to number of cores:
            num_workers = num_cores

        # Initialize a dict for sharing information
        # among the processes (other than assignments,
        # which are passed through queues):
        
        # Fill the inter-process list with False
        # to indicate none of the jobs has finished.
        # Will be used to avoid logging that fact that a
        # particular job finishing over and over to the 
        # console (i.e. not used for functions other than 
        # reporting progress):
        
        for _i in range(num_workers):
            # NOTE: reportedly one cannot just set the passed-in
            #       list to [False]*num_workers, b/c 
            #       a regular python list won't be
            #       seen by other processes, even if
            #       embedded in a multiprocessing.manager.list
            #       instance. Instead, global_info['jobs_status']
            #       is a special type of list created by mp.manager:
            global_info['jobs_status'].append(False)

        # Number of full spectrograms to chop:
        global_info['snips_to_do'] = len(Utils.find_in_dir_tree(spectrogram_locs, 
                                                                pattern="*.png")) 
        
        
        # For progress reports, get number of already
        # existing .png files in out directory:
        global_info['snips_done'] = len(Utils.find_in_dir_tree(outdir, 
                                                               pattern="*.png")) 

        # Find number of tasks to do: the number of snippets
        # any of the recordings in which we are interested:
        num_matching_tasks = self.count_recordings_by_rec_ids(
            spectrogram_locs, 
            list(rec_id_assocs.keys())
            )
        
        # Queue to pass snippet paths to workers.
        # All workers will grab snippet paths from the
        # queue, and try to match them against selections:
        task_queue = mp.Manager().Queue(maxsize=self.TASK_QUEUE_SIZE)
        snip_matching_jobs = []

        # Start all the workers:
        for i in range(min(num_matching_tasks, num_workers)):
            ret_value_slot = mp.Value("b", False)
            job = mp.Process(target=self.process_one_sel_tbl,
                             args=(task_queue,),
                             name=f"snippet_matcher-{i}"
                             )
            job.ret_val = ret_value_slot
            snip_matching_jobs.append(job)
            print(f"Starting matching for table {job.name}")
            job.start()

        # Feed one snippet after the other 
        # to the task_queue, for workers to match
        # against selections:
        for assoc_count, recording_snippet_assoc in enumerate(rec_id_assocs.values()):
            # Each assoc is an instance containing
            # the path to a selection table (raven_sel_tbl_path),
            # and the recording ID (rec_id) associated with
            # that table. Additionally, the assoc is itself
            # a generator of snippet metadata for snippets
            # whose filename marks it as part of the recording
            # of ID rec_id. Feed all the metadata to the queue,
            # then loop to feed snippets for the next selection
            # table:
            
            for snippet_md in recording_snippet_assoc:
                # Create another dict from the snippet metadata
                # to serve as task argument for one worker:
                task_args = snippet_md.copy()
                # Add the selection table path to the md:
                task_args['sel_tbl_path'] = sel_tbl_paths[recording_snippet_assoc.rec_id]
                task_args['outdir'] = outdir
                # Among other info, snippet_md now has:
                # {... 'start_time(secs)': '2.0',
                #      'end_time(secs)': '6.9',
                #      'snip_path': 'AM01_20190711_170000_....png',
                #      'sel_tbl_path': 'DS_AM01_20190711_170000...txt',
                #      'outdir' : '/foo/bar/labeled_snippets/',
                #      ...}
                # Pass that dict into the task queue to be picked
                # up by any of the workers:
                task_queue.put(task_args)
                if assoc_count > 0 and assoc_count % 1000 == 0:
                    # Report progress in place, w/o CR ("end='")
                    if self.log.logging_level >= INFO:
                        print(f"Matching jobs submitted: {assoc_count}",
                              end=''
                              )

        # All tasks have been fed into the task
        # queue (though they may not be done yet).
        # Feed one 'poison' entry into the queue for
        # each worker to swallow and terminate:
        for _i in range(num_workers):
            task_queue.put('STOP')
        #start_time = datetime.datetime.now()

        # Keep checking on each job, until
        # all are done as indicated by all jobs_done
        # values being True, a.k.a valued 1:
        
        while sum(global_info['jobs_status']) < num_workers:
            for job_idx, job in enumerate(snip_matching_jobs):
                # Timeout 1 sec
                job.join(4)
                if job.exitcode is not None:
                    if global_info['jobs_status'][job_idx]:
                        # One of the processes has already
                        # reported this job as done. Don't
                        # report it again:
                        continue
                        
                    # Let other processes know that this job
                    # is done, and they don't need to report 
                    # that fact: we'll do it here below:
                    global_info['jobs_status'][job_idx] = True
                    
                    # This job finished, and that fact has not
                    # been logged yet to the console:
                    
                    res = "OK" if job.ret_val else "Error"
                    # New line after the single-line progress msgs:
                    print("")
                    print(f"Worker {job.name}/{num_workers} finished with: {res}")
                    #global_info['snips_done'] = self.sign_of_life(job, 
                    #                                              global_info['snips_done'],
                    #                                              outdir,
                    #                                              start_time,
                    #                                              force_rewrite=True)
                    # Check on next job:
                    continue
                    #
                # # This job not finished yet.
                # # Time for sign of life?
                # global_info['snips_done'] = self.sign_of_life(job, 
                #                                              # global_info['snips_done'],
                #                                              # outdir,
                #                                              # start_time,
                #                                              # force_rewrite=True)

    #------------------------------------
    # process_one_sel_tbl 
    #-------------------
    
    def process_one_sel_tbl(self, task_queue):
        '''
        This is the target of each process once it
        is forked by the multiprocess.Process(...).start()
        call. 
        
        Thus each new instance variable will be unique
        to one (i.e. this) process. Similarly, the 
        'name' instance var will be a human readable string
        naming this process for purposes of logging.
        
        The task queue is a multiprocessing.Queue. Each item
        fed from the work coordinator to this process via the
        queue is a dict containing the following:

           *******FILL IN 
           
        Each task is to match one snippet to any of the selections
        in the associated selection table. If a match is found,
        the snippet's metadata is augmented with the species label,
        and placed into the outdir. Else the snippet is labeled as
        'noise' and also placed into the outdir.
        
        When the snippet is processed, task_queue.task_done() is called,
        and the next item is taken from the queue.
        
        When the string 'STOP' is passed through the queue, this
        process calls task_queue.task_done() one last time, and terminates.

        :param task_queue: queue through which work coordinator 
            feeds tasks.
        :type task_queue: mp.queue
        '''

        my_job_name = mp.current_process().name
        # Dict mapping selection table path to 
        # the list of dicts that represent the selection
        # table rows:
        
        selections_all_tables = {}
        start_time = datetime.datetime.now()
        num_snips_processed = 0

        done = False
        while not done:
            
            task_args = task_queue.get()
            try:
                if task_args == 'STOP':
                    done = True
                    continue
                
                sel_tbl_path = task_args['sel_tbl_path']
                try:
                    selections = selections_all_tables[sel_tbl_path]
                except KeyError:
                    # Haven't encountered this sel table yet:
                    self.log.info(f"Reading selection table {sel_tbl_path}...")
                    selections = Utils.read_raven_selection_table(sel_tbl_path)
                    selections_all_tables[sel_tbl_path] = selections
    
                # If this snippet matches any selection in one of
                # the cached selection tables, enrich the snippet's 
                # metadata with species info. Then copy the enriched
                # snippet to the target dir:

                outdir = task_args['outdir']
                    
                self.match_snippet(selections, task_args, outdir)
                num_snips_processed += 1
                
                self.sign_of_life_progress_provided(my_job_name, 
                                                    num_snips_processed, 
                                                    start_time, 
                                                    force_rewrite=True)
            finally:
                task_queue.task_done()

    #------------------------------------
    # match_snippet
    #-------------------
    
    def match_snippet(self, selections, snip_metadata, outdir):
        '''
        Workhorse:
        For each snippet_path S, examines the time span covered by S.
        Finds the selection table row (if any) whose begin/end
        times overlap with S's time span.
        
        Modifies the snippet's metadata by adding:
        
         'low_freq'          : <low bound of frequency involved in vocalization
         'high_freq'         : <high bound of frequency involved in vocalization
         'multiple_species'  : <list of species heard simultaneously>
         'type'              : <whether Song/Call/Call-1/Call-Trill...>
        
        The selections are expected to be a list of dicts as
        produced by Utils.read_raven_selection_table(). This
        means:
        
           o Dicts are sorted by selection start time
           o the d['mix'] is a possibly empty list of species
                names.

        :param selections: list of dicts, each containing the 
            information of one selection table row. 
        :type selections: {str : str}
        :param snip_metadata: metadata of one snippet, with 'snip_path'
            entry added
        :type snip_metadata: {str : Any}
        :param outdir: directory where to write the updated
            snippets. Value is allowed to be same as snippet_path,
            but then the snippets will be updated in place
        :type outdir: str
        '''

        snippet_path = snip_metadata['snip_path']
        # This is the second time the snippet is loaded!!!!
        # First time was in method snips_iterator() of
        # class SelTblSnipsAssoc. See TODO at top for need
        # to avoid this time consuming operation:
        spectro_arr, metadata = SoundProcessor.load_spectrogram(snippet_path)
        
        # Sanity check:
        try:
            snippet_tstart = float(metadata['start_time(secs)'])
        except Exception as e:
            print(e)

        # End time of snippet:
        snippet_tend = float(metadata['end_time(secs)'])
    
        # Find the index of the select table row (dict)
        # whose time interval overlaps:

        snippet_interval = Interval(snippet_tstart, snippet_tend)
        
        # The returned selection dict *may* be a 
        # 'phantom' selection, which is created in 
        # find_covering_sel() when a snippet straddles 
        # multiple selection rows:
        
        sel_dict = self.find_covering_sel(selections, snippet_interval)
        
        if sel_dict is None:
            # This snippet_path was not involved in
            # any of the human-created selection
            # rectangles. ******* JUST RETURN IF MIX IS NON-EMPTY?
            metadata['species'] = 'noise'
            self.save_updated_snippet(outdir, 
                                      'noise', 
                                      Path(snippet_path).name, # without parents 
                                      spectro_arr, 
                                      metadata)
            return

        low_f     = sel_dict['Low Freq (Hz)']
        high_f    = sel_dict['High Freq (Hz)']
        species   = sel_dict['species']
        voc_type  = sel_dict['type'] # Song/Call/Song-Trill, etc.
        # Get possibly empty list of species
        # names that also occur in the selection:
        multiple_species = sel_dict['mix']
        
        # The species, and entries in the mix field
        # will be used as part of file names.
        # So ensure that they have no spaces.
        # Also: convert the "no bird" entries to
        # 'noise':
        
        if species == 'no bird':
            species = 'noise'
        else:
            species = species.replace(' ', '_')
        new_multiple_species = []
        for entry in multiple_species:
            if entry == 'no bird' or entry == 'noise':
                # Don't add noise as "also-present":
                continue
            else:
                new_multiple_species.append(entry.replace(' ', '_'))

        metadata['species'] = species
        metadata['low_freq'] = low_f
        metadata['high_freq'] = high_f
        metadata['type'] = voc_type
        metadata['multiple_species'] = new_multiple_species

        # If this snippet is marked as noise,
        # but the multiple_species field indicates
        # that the snippet juts into non-noise 
        # selections, don't save this snippet as noise
        # right here, but allow the loop over the
        # muiltiple_species below to save it as a
        # non-noise snippet; the second clause of the
        # 'or' says: this snippet truly is noise:
        
        if species != 'noise' or \
            (species == 'noise' and len(new_multiple_species) == 0):
            self.save_updated_snippet(outdir, 
                                      species, 
                                      Path(snippet_path).name, # without parents 
                                      spectro_arr, 
                                      metadata)

        # If the snippet_path matched, and contained multiple
        # overlapping calls, create a copy of the snippet_path
        # for each species:
        if len(new_multiple_species) > 0:
            # Ensure the human coder did not include
            # the primary species in the list of overlaps:
            try:
                del new_multiple_species[new_multiple_species.index(species)]
            except (IndexError, ValueError):
                # All good, species wasn't in the list of additionals
                pass
            for overlap_species in new_multiple_species:
                # If this snippet reaches into a selection
                # that simply records "no bird" or "noise",
                # no need to create a phantom, b/c noise
                # is everywhere anyway:
                if overlap_species == 'no bird' or \
                    overlap_species == 'noise':
                    continue
                metadata['species'] = overlap_species
                # New name for a copy of this snippet_path:
                p = Path(snippet_path)
                new_fname = f"{p.stem}_{overlap_species}{p.suffix}"
                self.save_updated_snippet(outdir, 
                                          overlap_species,
                                          new_fname,
                                          spectro_arr, 
                                          metadata)

    #------------------------------------
    # find_covering_sel
    #-------------------

    def find_covering_sel(self, sels, time_interval):
        '''
        Given a time interval and a list of Raven
        select table row dicts, return the dict that
        contains at least part of the interval. If the
        time interval does not lie at least partially
        within any selection, returns None. 
        
        Several cases:
        
        
        Selections:
        
                    |-----|        |-----|   |-------|
        Case1: xxx
        Case2    xxxxxx
        Case3          xxx
        Case4           xxxxx
        Case5                 xxx
        Case6                           xxxxxxxx
        Case7                                            xxxx
        Case8                     xxxxxxxxxx

        where xxx is the given time interval, and |---| is one
        of the selections

               
        :param sels: list of information contained in selection 
            table rows
        :type sels: [{str : str}]
        :param time_interval: a time interval in seconds
        :type time_interval: Interval
        :return: the selection that at least partially overlaps
            with the given time interval
        :rtype: {None | {str : str}}
        '''
        time_low  = time_interval['low_val']
        time_high = time_interval['high_val']
        
        # Line all the selections' begin times
        # up into a list. That makes it easy
        # to use the bisect package:
        begin_times = [sel['Begin Time (s)']
                       for sel
                       in sels]
        # Same for all labeling end times
        end_times   = [sel['End Time (s)']
                       for sel
                       in sels]
        end_times_desc = end_times.copy()
        end_times_desc.reverse()

        # Case 1? Interval entirely to left
        # of left-most selection
        if begin_times[0] > time_high:
            return None
        # Case 7? Interval entirely to right
        # of right_most selection
        if end_times[-1] <= time_low:
            return None

        # Check for other cases 2-6,8:
        try:
            
            # Get smallest sel whose begin is below time_low,
            # and whose end_time is higher than time_low
            # i.e. left-most sel that contains time_low:
            found = False
            start_idx = 0
            while not found:
                try:
                    sel_left_idx, _left_begin_sel_time  = self.find_le(begin_times[start_idx:], 
                                                                           time_low)
                    if end_times[sel_left_idx] > time_low:
                        found = True
                    else:
                        start_idx = sel_left_idx + 1
                        if start_idx >= len(begin_times):
                            # No selection is covered by the snippet:
                            return None
                except ValueError:
                    # Snippet begins before the currently still possible
                    # selections. Does it reach into the the first of those
                    # selections, or maybe even beyond?
                    if time_high >= begin_times[start_idx:][0]:
                        sel_left_idx = start_idx
                        found = True
                        continue
                    
                    # Snippet is earlier than even the left-most 
                    # of all selections
                    return None

            # Get the first sel whose end is greater than time high,
            # i.e. the right-most candidate sel whose end point is above
            # the snippet:
            sel_right_idx, _right_end_sel_time = self.find_gt(end_times, time_high)

            # If this entire snippet gt the right-most interval?
            if begin_times[sel_right_idx] > time_high: #**************
                sel_right_idx -= 1

            left_sel = sels[sel_left_idx]
            
            # We did not draw a ValueError, so
            # time_low and time_high are both inside
            # the left-most and right-most bound of all
            # selections.
            
            # If the two indices are the same, 
            # the interval is entirely contained
            # in one selection:
            if sel_left_idx == sel_right_idx:
                # Case 3: containment:
                return sels[sel_left_idx]

            # How many selections are covered by the snippet?
            
            sel_idxes_involved = 1 + sel_right_idx - sel_left_idx
            
            # Strategy: create a new 'phantom' 
            # selection that includes all species
            # in the 'mix' and 'species' fields.
            # The caller will replicate this
            # snippet to account for all the
            # species: 

            other_species = set()
            for sel_idx_other in range(sel_left_idx,
                                       sel_left_idx + sel_idxes_involved
                                       ):
                other_sel = sels[sel_idx_other]
                if other_sel['mix'] is not None and len(other_sel['mix']) > 0:
                    # Combine all non-noise mix and species:  
                    other_sel_non_noise_species = [species
                                                   for species
                                                   in other_sel['mix'] + [other_sel['species']]
                                                   if species not in ['noise', 'no_bird', 'no bird']
                                                   ]
                    other_species  = set(other_species.union(other_sel_non_noise_species))
                elif other_sel['species'] not in ['noise', 'no_bird', 'no bird']:
                    other_species.add(other_sel['species'])

            # The new dict's species shouldn't be in
            # the 'mix' field, for which the copies
            # of the snippet will be made, because this
            # snippet will already receive the left-most
            # sel's data:
            try:
                other_species.remove(left_sel['species'])
            except KeyError:
                pass    
            
            other_species_lst = list(other_species)

            new_sel = left_sel.copy()
            # New dict's 'mix' is all species of the straddled
            # dicts (converting the set to a list):
            new_sel['mix'] = other_species_lst

            return new_sel
                
        except ValueError:
            pass

        # Case 2: only the end time lies within a selection:
        try:
            le_idx, _le_val = self.find_le(begin_times, time_high)
            return sels[le_idx]
        except ValueError: 
            pass
            
        # Case 4: only the start time is within a selection:
        try:
            le_idx, _le_val = self.find_le(end_times, time_low)
            return sels[le_idx]
        except ValueError: 
            pass

        return None

    #------------------------------------
    # find_le 
    #-------------------

    def find_le(self, a, x):
        'Find rightmost value less than or equal to x'
        i = bisect_right(a, x)
        if i:
            return (i-1, a[i-1])
        raise ValueError()

    #------------------------------------
    # find_gt 
    #-------------------

    def find_gt(self, a, x):
        'Find leftmost value greater than x'
        i = bisect_right(a, x)
        if i != len(a):
            return (i, a[i])
        raise ValueError()

    #------------------------------------
    # create_snips_gen_for_sel_tbls 
    #-------------------
    
    def create_snips_gen_for_sel_tbls(self, snippets_src, sel_tables_src):
        '''
        Given one or more Raven selection tables, 
        and one or more recording snippet paths, return
        a dict:
        
               {<recording-id> : SelTblSnipsAssoc-inst<table-path, snippets-dir>}

        where recording-id is like AM01_20190719_063242; table-path
        is the full path to one selection table with the respective
        recording-id, and snippets-dir is root of a director containing
        the snippets covered in the recording. 
        
        Usage concept:
            o There are relatively few selection tables, since they
              are human-generated
            o There can be thousands of snippet .png files whose time spans
              are covered in one table
            o The data structure returned from this method can be
              used like this:
              
                    tbl_snips_match = create_snips_gen_for_sel_tbls('/foo/my_snips', '/bar/my_tbls')
                    
                    # For each selection table, work on the snippets
                    # that are covered by that table
                    
                    for rec_id in tbl_snips_match:
                        for snip_path in tbl_snips_match.snips_iterator():
                            <do something with spectrogram snippet>
        
        
        :param snippets_src: iterable over absolute paths to snippets,
            or the absolute path to a directory
        :type snippets_src: {Iterator(str) | str}
        :param sel_tables_src: absolute path to selection table, or path 
            to a directory that contains selection tables, or
            iterator over absolute paths to selection tables
        :type sel_tables_src: str
        :returned dict mapping recording ID to SelTblSnipsAssoc instances
        :rtype {str : SelTblSnipsAssoc}
        '''

        # Table paths may be an individual
        # file, a directory, or a generator/iterator
        # of absolute paths. Sanity checks:
        
        if type(sel_tables_src) == str:
            if not os.path.isabs(sel_tables_src):
                raise ValueError(f"Table paths must be a generator, or an absolute path to a selection table or dir")
            if os.path.isfile(sel_tables_src):
                sel_tables_src = [sel_tables_src]
            elif os.path.isdir(sel_tables_src):
                sel_tables_src = Utils.listdir_abs(sel_tables_src)
        # If not a string, sel_tables_src better be a generator:
        elif not isinstance(sel_tables_src, Iterable):
            raise ValueError(f"Table paths must be a generator, or an absolute path to a selection table or dir")

        # Same checks for snippet location:
        if type(snippets_src) == str:
            if not os.path.isabs(snippets_src) \
                or not os.path.isdir(snippets_src):
                raise ValueError(f"Snippet paths must be a generator, or an absolute path to a snippet dir")
            snippets_src = iter(Utils.listdir_abs(snippets_src))
        # If not a string, snippets_src better be a generator:
        elif not isinstance(sel_tables_src, Iterable):
            raise ValueError(f"Snippets src must be a generator, or an absolute path to dir")

        # Build a dict:
        #    {<recording_id> : <dir-of-matching-snippets>}
        recording_selection_tables = {}
        for table_path in sel_tables_src:
            recording_id = self.extract_recording_id(table_path)
            if recording_id is not None:
                recording_selection_tables[recording_id] = \
                    SelTblSnipsAssoc(table_path, snippets_src)

        return recording_selection_tables

    #------------------------------------
    # extract_recording_id
    #-------------------
    
    @classmethod
    def extract_recording_id(cls, spectro_path_or_fname):
        '''
        Given any name like:
        
            o [<parent_dirs>]/AM01_20190719_063242.png
            o [<parent_dirs>]/DS_AM01_20190711_170000.Table.1.selections.txt
        
        return the portion:
               <recorder-id>_<date>_<recording_id> 
        
        where <recorder-id> is AMnn (AM: AudioMoth).

        :param spectro_path_or_fname:
        :type spectro_path_or_fname:
        :return: substring <recorder-id>_<date>_<recording_id>,
            or None if no match
        :rtype: str
        '''
        fname_only = Path(spectro_path_or_fname).stem
        m = cls.RECORDING_ID_PAT.search(fname_only)
        if m is not None:
            return m.groups()[0]
        return None

    #------------------------------------
    # save_updated_snippet
    #-------------------
    
    def save_updated_snippet(self, outdir, species, snippet_path, spectro_arr, metadata):
        '''
        Create path name: 
            
            outdir/species/snippet-fname
            
        and save the spectro_arr to that path
        as a .png file with embedded metadata
        
        :param outdir: destination directory
        :type outdir: str
        :param snippet_path: file name or absolute path to snipet
        :type snippet_path: src
        :param spectro_arr: image data
        :type spectro_arr: np.array
        :param metadata: auxiliary info to include in the .png file
        :type metadata: {str : str}
        '''
        
        # Save the updated snippet_path:
        species_subdir = os.path.join(outdir, species)
        snip_outname = os.path.join(species_subdir, os.path.basename(snippet_path))
        FileUtils.ensure_directory_existence(snip_outname)
        SoundProcessor.save_image(spectro_arr, snip_outname , metadata)

    #------------------------------------
    # make_sel_tbl_dict 
    #-------------------
    
    def make_sel_tbl_dict(self, sel_tbls_root):
        '''
        Given either the path to a single selection
        table, or to the root of a directory of selection
        tables, return a dict or recording ID to full
        selection table paths.
        
        Assumption: selection table file names are of the
        form 
             DS_AM02_20190716_172958.Table.1.selections.txt
               [--------------------]                 [----]
                   important part                    Required
                                                       suffix 
        
        The 'important part' is the recording ID, which will
        be used as the dict keys. Files not matching the above
        pattern will be ignored.
        
        :param sel_tbls_root: path to selection table, or to
            root directory of selection tables 
        :type sel_tbls_root: src
        :return: dict mapping recording ID to absolute paths
        :rtype: {src : src}
        '''
        pat = re.compile(re.compile(r'.*(AM[0-9]{2}_[0-9]{8}_[0-9]{6}).*[.]txt$'))
        res = {}
        
        # Individual selection table file?
        if os.path.isfile(sel_tbls_root):
            match = pat.match(sel_tbls_root)
            if match is not None:
                res[match.groups()[0]] = sel_tbls_root
            return res
        
        # Walk directory, collecting sel tbl files:
        for root, _dirs, files in os.walk(sel_tbls_root):
            for file in files:
                match = pat.match(file)
                if match is not None:
                    res[match.groups()[0]] = os.path.join(root, file)
        return res


    #------------------------------------
    # count_recordings_by_rec_ids
    #-------------------

    def count_recordings_by_rec_ids(self, root, rec_ids):
        '''
        Given a directory tree root and a list of recording
        IDs, return the number of .png files containing any of
        the IDs. A recording ID is of the form
        
            <recorder-id>_<date>_<recording_id>
            
        As in the first part of 'AM01_20190711_170000_sw-start26_unk1.png'
        
        :param root: directory tree root
        :type root: str
        :param rec_ids: list of recording IDs for which to
            count png files
        :type rec_ids: [str]
        :return: count of matching png files
        :rtype: int
        '''
        count = 0
        if os.path.isfile(root) and \
            SnippetSelectionTableMapper.extract_recording_id(root) in rec_ids:
            count += 1
        for _dir_root, _dirs, files in os.walk(root):
            for snip_path in files:
                if SnippetSelectionTableMapper.extract_recording_id(snip_path) in rec_ids:
                    count += 1
        return count

    #------------------------------------
    # sign_of_life_progress_provided 
    #-------------------
    
    @classmethod
    def sign_of_life_progress_provided(cls, 
                     job_name, 
                     num_already_present_imgs, 
                     start_time, 
                     force_rewrite=False):
        
        # Time for sign of life?
        #*****************
        #force_rewrite=True
        #*****************

        now_time = datetime.datetime.now()
        time_duration = now_time - start_time
        # Every 3 seconds, but at least 3:
        if force_rewrite \
           or (time_duration.seconds > 0 and time_duration.seconds % 3 == 0): 
            
            # A human readable duration st down to minutes:
            duration_str = Utils.time_delta_str(time_duration, granularity=4)

            # Keep printing number of done snippets in the same
            # terminal line:
            print(f"{job_name}---Number of spectros: {num_already_present_imgs} after {duration_str}",
                  end='\r')

    #------------------------------------
    # sign_of_life 
    #-------------------
    
    @classmethod
    def sign_of_life(cls, 
                     job, 
                     num_already_present_imgs, 
                     outdir, 
                     start_time, 
                     force_rewrite=False):
        
        # Time for sign of life?
        now_time = datetime.datetime.now()
        time_duration = now_time - start_time
        # Every 3 seconds, but at least 3:
        if force_rewrite \
           or (time_duration.seconds > 0 and time_duration.seconds % 3 == 0): 
            
            # A human readable duration st down to minutes:
            duration_str = Utils.time_delta_str(time_duration, granularity=4)

            # Get current and new spectro imgs in outdir:
            num_now_present_imgs = len(Utils.find_in_dir_tree(outdir, pattern="*.png"))
            num_newly_present_imgs = num_now_present_imgs - num_already_present_imgs

            # Keep printing number of done snippets in the same
            # terminal line:
            print((f"{job.name}---Number of spectros: {num_now_present_imgs} "
                   f"({num_newly_present_imgs} new) after {duration_str}"),
                  end='\r')
            return num_newly_present_imgs
        else:
            return num_already_present_imgs


# ---------------------- Class SelTblSnipsAssoc ---------

class SelTblSnipsAssoc:
    '''
    Conceptually maintains a 1->many association:
    
         <raven-selection-tbl> ---> (metadata-of-covered-snippet1.png,
                                     metadata-of-covered-snippet2.png,
                                     metadata-of-covered-snippet3.png,
                                     )

    Where raven_sel_tbl_path is the absolute path to
    one Raven selection table. The members in the list 
    (i.e. metadata-of-covered-snippet<n>.png) are dicts
    of metadata extracted from the respective snippet .png file.
    An additional entry {'snip_path' : <absolute-path-to-snippet>} is
    added to the .png metadata.
    
    Instances act as generators that feed out metadata
    for snippets in order of start time.

    Since there may be many snippets covered
    in one Raven table, and many unrelated snippets may 
    reside in one directory, only snippets associated with
    recordings covered in the given selection table or processed.
    evaluation. Example use:
    
         # Create an instance of this class:
         tbl_snip_assoc = SelTblSnipsAssoc(my_sel_tbl.txt, my_snips_dir)
         
         # Print only snippet paths of snippets
         # covered by this instance's Raven selection table:
         
         for snip_metadata in tbl_snip_assoc:
             print(snip_metadata['snip_path'])
             
    the full list of snippets in my_snips_dir is never
    materialized.
    
    Note: 
    '''

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, raven_sel_tbl_path, snips_src):
        
        if type(snips_src) == str and not os.path.isdir(snips_src):
            raise ValueError(f'Arg snips_src must be a directory or generator/iterable, not {snips_src}')

        if not  (isinstance(snips_src, Iterable) or \
                 inspect.isgeneratorfunction(snips_src)
                 ):
            raise ValueError(f'Arg snips_src must be a directory, not {snips_src}')

        self.raven_sel_tbl_path = raven_sel_tbl_path
        self.snip_dir = snips_src
        self.rec_id   = SnippetSelectionTableMapper.extract_recording_id(raven_sel_tbl_path)


    #------------------------------------
    # __len__ 
    #-------------------
    
    def __len__(self):
        
        try:
            return self.num_snippets
        except Exception:
            # Iterator has not initialized this number yet:
            return 0

    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        return self.snips_iterator(self.snip_dir)
    
    #------------------------------------
    # snips_iterator
    #-------------------
    
    def snips_iterator(self, root=None):
        '''
        
        If root is a generator or iterable, we assume that it
        will yield absolute paths to snippets. If root 
        is None, we set root to self.snip_dir.
        Else, assume root is a directory below which snippets
        are found. 
        
        In the latter case: recursively find snippet files, 
        starting with root. Yield one snippet file after 
        the other, if its recording ID matches this
        instance's Raven selection table's ID.
        
        :param root: file or directory with snippet files
        :type root: {root | str | types.GeneratorTypes}
        :return a series of full snippet file paths whose
            file names indicate that the snippets are covered
            in this instance's Raven table.
        :rtype str
        :raise StopIteration
        '''
        if root is None:
            root = self.snip_dir
        
        metadata_list   = []
        
        
        if type(root) != str and \
           (isinstance(root, types.GeneratorType) or \
            isinstance(root, Iterable)
            ):

            # Root is a ready made generator.
            snip_gen = root
        else:
            # Make a generator of snippets that originate
            # from the given recording (id):
            snip_gen = self.dir_tree_snip_gen(root)
            
        
        # Create the time sorted list of snippet
        # metadata:
        for snip_path in snip_gen:
            _img_arr, metadata = SoundProcessor.load_spectrogram(snip_path, to_nparray=False)
            metadata['snip_path'] = snip_path
            metadata_list.append(metadata)

        time_sorted_metadata = sorted(metadata_list,
                                      key=lambda md: md['start_time(secs)'])

        self.num_snippets = len(time_sorted_metadata)
        # Now keep feeding the list of sorted
        # metadata:
        # Since Python 3.7 generators raising 
        # StopIteration is no longer silently 
        # discarded, need to catch it ourselves:
        try:
            for metadata in time_sorted_metadata:
                yield metadata
            return
        except StopIteration:
            return

    #------------------------------------
    # dir_tree_snip_gen
    #-------------------
    
    def dir_tree_snip_gen(self, root):
        '''
        Generator feeding out absolute snippets paths from 
        directory given in root arg. Only snippets whose
        recorder ID matches this instance's recorder_id are
        returned. 
        :param root: path to individual file, or to directory 
        :type root: str
        :returns a generator
        :rtype: types.GeneratorType
        '''

        # Feed out a snippet from dir tree:
        if os.path.isfile(root) and self.recording_id_matches(root):
            yield root
        for dir_root, _dirs, files in os.walk(root):
            for snip_path in files:
                if self.recording_id_matches(snip_path):
                    yield os.path.join(dir_root, snip_path)


    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        return f"<SelTblSnipsAssoc {self.rec_id} at {hex(id(self))}>"

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        return f"<SelTblSnipsAssoc at {hex(id(self))}>"

    #------------------------------------
    # recording_id_matches 
    #-------------------
    
    def recording_id_matches(self, snippet_path):
        snip_rec_id = SnippetSelectionTableMapper.extract_recording_id(snippet_path)
        return snip_rec_id == self.rec_id

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Match spectrogram snippets to Raven selection table."
                                     )

    parser.add_argument('selection_table_loc',
                        help='Path to Raven selection table(s): either to .csv file or parent directory')

    parser.add_argument('snippets_path',
                        help='Path directory of snippets with embedded time information')
    parser.add_argument('outdir',
                        help='Path directory where modified snippets will be placed (created if not exists)')

    args = parser.parse_args()
    
    # Multiprocessing the matching:
    # For nice progress reports, create a shared
    # dict quantities that each process will report
    # on to the console as progress indications:
    #    o Which job-completions have already been reported
    #      to the console (a list)
    #    o The most recent number of already created
    #      output images 
    # The dict will be initialized in __init__(),
    # but all Python processes on the various cores
    # will have read/write access:
    manager = mp.Manager()
    global_info = manager.dict()
    global_info['jobs_status'] = manager.list()

    matcher = SnippetSelectionTableMapper(args.selection_table_loc, 
                                          args.snippets_path,
                                          args.outdir,
                                          global_info
                                          )
