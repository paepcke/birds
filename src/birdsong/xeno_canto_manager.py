#!/usr/bin/env python3
"""
Download and manage Xeno Canto recordings.

Usage example:
   
   xeno_canto_manager.py --download "Tangara gyrola"
   

Functions involved in downloading birds from xeno-canto
Can be used from the command line (see argparse entries
at end of file). Or one can import XenoCantoCollection
and XenoCantoRecording into an application.

Two classes:
  
  o XenoCantoRecording to hold metadata about available XC
        sound files
  o XenoCantoCollection to hold XenoCantoRecording instances

A XenoCantoCollection instance acts like a dict whose keys are
concatenations of genus and species names, e.g. 'Tangaragyrola'
Each value in the dict is a *list* of XenoCantoRecording instances 
for that type of bird.

So the keys of a XenoCantoCollection instance might look like:

{'Tangaragyrola' : [rec1, rec2, rec3]
 'Amaziliadecora': [rec4, rec5]
}

A XenoCantoRecording also behaves like a dict:

Keys are:
        genus
        species
        phylo_name       # 'Tangaragyrola'
        full_name        # Unique name like 'Tangaragyrola_xc3562156' 
        country
        loc              # Name of park/city/preserve 
        recording_date
        length
        encoding         # ['mpeg', 'wav.vpn']
        type             # ['call', 'song']

        _xeno_canto_id
        _filename (if downloaded)
        _url
        
A XenoCantoCollection instance can also act as an iterator:

	 for recording in my_coll(one_per_bird_phylo=True):
	     print(recording.full_name)

The one_per_bird_phylo kwarg controls whether
only one of each species' recordings are included
in the iteration, or whether the list of all recordings 
for all species are served out.

The interaction model is to pull metadata for each
desired species by running this script from the command
line either with or without downloading the actual sound 
files. That process creates a XenoCantoCollection, which
is saved to disk as a pickle.

One can ask the XenoCantoCollection instance to
download some or all recordings. Individual XenoCantoRecording
instances also know how to download.

Working with the collection:

Clients can import the module, rather than calling
from the command line. The resource can therefore
be integrated into a workflow. 

After the sound files are downloaded, the following
is a template for use without re-contacting the Xeno Canto
server:

from birdsong.xeno_canto_manager import XenoCantoCollection, XenoCantoRecording
sound_collection = XenoCantoCollection.load('<path to pickled collection')

# The following iterates through
# XenoCantoRecording instances, *one*
# recording of each species:

for recording in sound_collection:
    print(recording.full_name)
    print(recording.filepath)

# The same, but all recordings of
# every species:

for recording in sound_collection(one_per_bird_phylo=False):
    print(recording.full_name)
    print(recording.filepath)

# TODO:
#   o Example for getting metadata

"""

# Use orjson for reading.
# It is lightning fast:
# TODO:
#   o Example for creating a .json collection
'''
Usage example assuming you have a 
  .json file with a previously saved collection

	cd ~/birds
	conda activate birds
	export PYTHONPATH=src:$PYTHONPATH# For the tests below, start with no# downloaded sound files:
	rm /home/yatagait/birds/src/birdsong/recordings/*
	
	python3
	# >>>
	import birdsong.xeno_canto_manager as xcm
	sound_collection = xcm.XenoCantoCollection.load('~/test_metadata.json')
	for rec in sound_collection:
	    print(rec)
	
	# Get messages about downloading and saving# as it rattles through:
	for rec in sound_collection:
	    dest_file = rec.download()
	
	# Download again: get warning about file
	# present. In this case, answer 'no' to the overwrite
	# question to avoid re-downloading any
	# of the already present files. 
	
	for rec in sound_collection:
	    dest_file = rec.download()
	
	# Download the collection all at once,
	# rather than in a loop. This won't
	# re-download, b/c it remembers your
	# 'no' answer from above:
	
	sound_collection.download()
	
	# Force re-downloading:
	sound_collection.download(overwrite_existing=True)

'''
# ---------------- Class XenoCantoCollection ----------

import argparse
import datetime
import os
from pathlib import Path
import pickle
import re
import sys
import time

from logging_service import LoggingService
import orjson
import requests

from birdsong.utils.species_name_converter import SpeciesNameConverter, \
    DIRECTION
from birdsong.utils.utilities import FileUtils
import numpy as np


class XenoCantoCollection:
    '''
    Holds information about multiple Xeno Canto
    recordings. Each holding is a XenoCantoRecordingO
    instance, keyed by species name. For each species
    may hold multiple recording instances.
    
    Constructor takes a raw download from Xeno Canto
    (see xeno_canto_get_bird_metadata()). Downloads are
    of the form:
    
             {<download metadata: num recordings, num species,...
                "recordings" : [{recording metadata: recording date,
                                                     species name,
                                                     download info,
                                                     ...
                                },
                                {next recording metadata},
                                ...
                               ]

    '''
    # By default, iterating through a list
    # of XenoCantoRecording instances that
    # are of the same bird, just different
    # recordings, only return the first of
    # each species' recordings, then move
    # on to the next:
    #
    #     sound_collection: species1: [rec_s1_1, rec_s1_2,...],
    #           species2: [rec_s2_1, rec_s2_2,...],
    #
    # If one_per_bird_phylo is True, then 
    # an iterator over the collection will return
    #
    #     rec_s1_1, rec_s2_1
    #
    # Else each recording of each species is 
    # fed out:
    
    one_per_bird_phylo = True 


    #------------------------------------
    # __new__ 
    #-------------------
    
    def __new__(cls, bird_names_or_coll_path, **kwargs):
        '''
        If bird_names_or_coll_path is the path
        to a previously saved collection, then
        instantiate from that file, and return
        that instance.
        
        Else if bird_names_or_coll_path is a list
        of birds, a XenoCantoCollection is created,
        and initialized with downloaded metadata
        I.e. the metadata download happens as part of 
        instantiation.
        
        :param bird_names_or_coll_path: path to saved
            XenoCantoCollection instance, or list of
            bird species.
        :type bird_names_or_coll_path: {str | list[str]}
        '''
        
        # Initialize all data structures based on 
        # Either a passed-in path to a previously 
        # saved XenoCantoCollection instance, or on
        # metadata downloads.
        
        if bird_names_or_coll_path is None:
            # Being called during reading of
            # a saved XenoCantoCollection instance.
            # Just make an empty instance, return it,
            # and let the load function fill all the
            # inst vars. The __init__ method will be
            # called, though the load method might overwrite
            # some of the inst vars that are initialized
            # there:
            
            inst = super().__new__(cls, **kwargs)
            inst.data = {}
            return inst
        
        if type(bird_names_or_coll_path) == list:
            # Make a fresh instance as normal
            #*****inst = super().__new__(cls, **kwargs)
            inst = super().__new__(cls)
            inst.__init__(**kwargs)
            inst.data = {}
            # Download metadata, and create XenoCantoRecording
            # instances:
            inst.instantiate_recs(bird_names_or_coll_path)
            return inst
            
        else:
            # Note: load() will instantiate a XenoCantoCollection,
            # recursively calling this __new__ method with None. 
            # That is what is taken care of above:
            inst = cls.load(bird_names_or_coll_path)
            return inst

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 courtesy_delay=1.0,
                 dest_dir=None,
                 always_overwrite=True):
        '''
        Given either a list of bird names to 
        download from xeno canto, or the
        path to a previously saved XenoCantoCollection
        instance, instantiate a XenoCantoCollection
        instance.
        
        The reason for defaulting always_overwrite
        to True is that occasionally Xeno Xanto
        delivers a file twices. Thus, if caller forgets
        to set always_overwrite to True, a long
        download will occasionally ask permission to
        overwrite. This makes it impossible to run
        a long download unattended.
        
        :param bird_names_or_coll_path: path to 
            previously saved collection (a .json, 
            or .pkl/.pickle file. Or the list of
            bird species as needed to query Xeno Canto
        :type bird_names_or_coll_path: {list | str|
        :param courtesy_delay: seconds between downloads
            to be polite to Xeno Canto server. Don't
            set this to zero! You may well be blocked
            from the site.
        :type courtesy_delay: float
        :param dest_dir: destination of any recording
            downloads. Default: subdirectory 'recordings'
            under this script's directory
        :type dest_dir: str
        :param always_overwrite: during download, 
            overwrite any already existing recordings 
            with new one without asking.
        :type always_overwrite: bool
        '''
        # No cached "number of recordings" yet:
        self._num_recordings = None
        
        self.log = LoggingService()
        self.always_overwrite = always_overwrite
        if always_overwrite:
            self.log.info("Note: if dowloads requested, will overwrite any existing recordings.")
        
        self.courtesy_delay = courtesy_delay
        
        curr_dir = os.path.dirname(__file__)
        if dest_dir is None:
            self.dest_dir = os.path.join(curr_dir, 'recordings')
            if not os.path.exists(self.dest_dir):
                os.mkdir(self.dest_dir)
        else:
            self.dest_dir = dest_dir
            
        # Default behavior of iterator (see comment in __iter__):
        self.one_per_bird_phylo=True
    
    #------------------------------------
    # instantiate_recs 
    #-------------------
    
    def instantiate_recs(self, bird_names):
        '''
        Given a list of bird species, download
        each bird's metadata. Then create a
        XenoCantoRecording instance for each
        recording. Don't download the recordings
        themselves. 
        
        :param bird_names: list of species as needed
            to download from the Xeno Canto server
        :type bird_names: [str]
        '''
        collections_metadata = self.xeno_canto_get_bird_metadata(bird_names)

        # Create XenoCantoRecording instances
        # for each recording mentioned in the
        # metadata:
        for species_recs_metadata in collections_metadata:
            # Metadata about XC holdings for one genus/species:
            self.num_species = species_recs_metadata['numSpecies']
            # List of dicts, each describing one 
            # recording:
            recordings_list  = species_recs_metadata['recordings']
    
            # Create XenoCantoRecording instances
            # without sound file download. Downloading is
            # done lazily:
            
            for rec_dict in recordings_list:
                # Add the 4-letter and 6-letter codes
                # to the recording's metadata:
                sci_name = f"{rec_dict['gen']}_{rec_dict['sp']}"
                try:
                    rec_dict['four_code'] = SpeciesNameConverter()[sci_name,
                                                                   DIRECTION.SCI_FOUR]
                except KeyError:
                    # Could not find a four-letter code from
                    # the scientific name:
                    rec_dict['four_code'] = None
                    
                try:
                    rec_dict['six_code'] = SpeciesNameConverter()[sci_name,
                                                                   DIRECTION.SCI_SIX]
                except KeyError:
                    # Could not find a six-letter code from
                    # the scientific name:
                    rec_dict['six_code'] = None

                rec_obj  = XenoCantoRecording(rec_dict, 
                                              dest_dir=self.dest_dir, 
                                              log=self.log)
                try:
                    phylo_name = rec_obj.phylo_name
                    self[phylo_name].append(rec_obj)
                except KeyError:
                    # First time seen this species:
                    self[phylo_name] = [rec_obj]

    #------------------------------------
    # xeno_canto_get_bird_metadata 
    #-------------------

    def xeno_canto_get_bird_metadata(self, bird_names):
        num_to_get = len(bird_names)
        metadata = []
        for bird_name in bird_names:
            
            url = "https://www.xeno-canto.org/api/2/recordings"
            http_query = f"{url}?query={bird_name}+len:5-60+q_gt:C"
            
            self.log.info(f"Downloading metadata #{len(metadata) + 1} / {num_to_get} bird(s)...")
            try:
                response = requests.get(http_query)
            except Exception as e:
                self.log.err(f"Failed to download metadata for {bird_name}: {repr(e)}")
                time.sleep(self.courtesy_delay)
                continue
            self.log.info(f"Done download")
            
            time.sleep(self.courtesy_delay)
            
            try:
                metadata.append(response.json())
            except Exception as e:
                self.log.err(f"Error reading JSON version of response: {repr(e)}")
                somewhat_readable = re.sub(r'<[^>]+>', '', str(response.content))
                self.log.err(somewhat_readable)
                continue
        return metadata

    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        
        # Index into the list of this collection
        # instance's keys. Each key's value is a list
        # of recording instances:
        self.curr_bird_phylo_indx = 0
        
        # Indx into one of the recording
        # lists. Used in case of one_per_bird_phylo
        # is False.
        self.curr_recording_indx = -1
        
        self.phylo_name_list = list(self.keys())

        return self
    
    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        '''
        Looks worse than it is. Recall:
        'self' is a dict:
           {
              phylo_name1 : (recording_instance1, recording_instance2,...) 
              phylo_name2 : (recording_instance1, recording_instance2,...) 
               ...
            }
        
        If iterator is to feed only one recording for each
        phylo_name, track phylo_names with self.curr_bird_phylo_indx
        into the list of phylo_name_n. The 'next' recording is
        the first recording in the next phylo_name.
        
        If iterator is to feed every bird, two indices are needed
        the one into phylo_names, and one into the current phylo_name
        value's list of recordings: self.curr_recording_indx. For
        each phylo_name we feed one recording after the other,  until
        the list is exhausted, then move on to the next phylo_name.
        
        :return: Xeno Canto recording instance
        :rtype: XenoCantoRecording
        '''
        if not self.one_per_bird_phylo:
            curr_phylo_name = self.phylo_name_list[self.curr_bird_phylo_indx]
            curr_recording_list = self[curr_phylo_name]
            # We are to feed all recordings
            # of all pylos:
            self.curr_recording_indx += 1
            try:
                return curr_recording_list[self.curr_recording_indx]
            except IndexError:
                # Fed out all recordings in current list
                # Move on to the next phylo, first resetting
                # the pointer into the list:
                self.curr_recording_indx = 0
                self.curr_bird_phylo_indx +=1
            except Exception as e:
                # Unexpected error
                raise RuntimeError(f"During next in collection: {repr(e)}") from e
                 
        # On to the next phylo entry:
        
        try:
            # Try to get next phylo name. If 
            # fails, we fed out everything:
            next_phylo_name = self.phylo_name_list[self.curr_bird_phylo_indx]
            nxt_rec_list = self[next_phylo_name]
            if self.one_per_bird_phylo:
                self.curr_bird_phylo_indx += 1
        except IndexError:
            # Have fed all records of all phylos:
            # Restore default for one_per_bird_phylo:
            self.one_per_bird_phylo = True
            raise StopIteration()

        # Have a new phylo entry's recording list:
        # Degenerate case: empty recordings list:
        if len(nxt_rec_list) == 0:
            # Recursively get next:
            return self.__next__()

        # Return first recording of list:
        return nxt_rec_list[0] 


    #------------------------------------
    # __setitem__ 
    #-------------------
    
    def __setitem__(self, key, val):
        self.data[key] = val

    #------------------------------------
    # __getitem__ 
    #-------------------
    
    def __getitem__(self, key):
        return self.data[key]

    #------------------------------------
    # __delitem__ 
    #-------------------
    
    def __delitem__(self, key):
        del self.data[key]
    
    #------------------------------------
    # num_recordings 
    #---------------
    
    @property
    def num_recordings(self):
        '''
        Lazily evaluated num_recordings property.
        The quantity is the number of all recordings
        in this collection instance's values(). If
        those values are themselves collections, add
        up their number of recordings.


        '''
        if self._num_recordings is None:
            all_lengths = [len(recording_list) 
                           for recording_list 
                            in self.values()
                            ]
            self._num_recordings = sum(all_lengths)
        return self._num_recordings 

    #------------------------------------
    # keys 
    #-------------------
    
    def keys(self):
        return self.data.keys()

    #------------------------------------
    # values
    #-------------------
    
    def values(self):
        return self.data.values()
    
    #------------------------------------
    # items
    #-------------------
    
    def items(self):
        return self.data.items()
    
    #------------------------------------
    # all_recordings 
    #-------------------
    
    def all_recordings(self):
        '''
        Convenience method: returns a list
        of all XenoCantoRecording instances
        in this collection. I.e. not separated
        by species, just all recordings from all
        species appended together
        '''
        all_recs = np.concatenate(list(self.values())).tolist()
        return all_recs
        

    #------------------------------------
    # download 
    #-------------------
    
    def download(self, 
                 birds_to_process=None,
                 one_per_species=True, 
                 courtesy_delay=1.0,
                 overwrite_existing=None,
                 ):
        '''
        Download a given list of species, or
        all the species in the collection. 
        
        If one_per_species is True, only one recording
        from the list of each species' recordings is
        downloaded; else all recordings of each requested
        species.
        
        Courtesy delay is time to wait between sound file
        downloads.
        
        :param birds_to_process: list of bird species names
        :type birds_to_process: [str]
        :param one_per_species: whether or not to download all 
            recordings of each species, or just one
        :type one_per_species: bool
        :param courtesy_delay: time between requests to XC server
        :type courtesy_delay: {int | float}
        :param overwrite_existing: if True, already existing 
            sound files will be overwritten without asking.
            If False, always ask once, then use answer as 
            default going forward. If None: same as False.
        :type overwrite_existing: bool
        '''
        
        # If file overwrite behavior is specified,
        # make it the default for all the XenoCantoRecording
        # instances:
        if overwrite_existing is not None:
            XenoCantoRecording.always_overwrite = overwrite_existing
        
        if birds_to_process is None:
            birds_to_process = list(self.keys())

        # Get number of sound files to download:
        if one_per_species:
            num_to_download = len(birds_to_process)
        else:
            num_to_download = 0
            for species in birds_to_process:
                try:
                    num_to_download += len(self[species])
                except KeyError:
                    self.log.err(f"Species {species} not represented in collection")

        downloaded = 1
        for rec in self(one_per_bird_phylo=one_per_species):
            self.log.info(f"{downloaded}/{num_to_download}")
            rec.download()
            downloaded += 1
            time.sleep(courtesy_delay)

    #------------------------------------
    # __len__ 
    #-------------------
    
    def __len__(self):
        '''
        Returns number of genus-species are in the
        collection. But does not sum up all the 
        recordings for each genus-species. To get
        that total number, use len_all()
        '''
        return len(self.data)

    #------------------------------------
    # len_all 
    #-------------------
    
    def len_all(self):
        num_recordings = 0
        for recs_list in self.data.values():
            num_recordings += len(recs_list)
        return num_recordings

    #------------------------------------
    # __eq__ 
    #-------------------
    
    def __eq__(self, other_coll):
        
        for phylo_name, rec_obj_list in self.items():
            other_rec_obj_list = other_coll[phylo_name]
            for rec_obj_this, rec_obj_other in zip(rec_obj_list, other_rec_obj_list):
                if rec_obj_this != rec_obj_other:
                    return False
        return True

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        #return f"<XenoCantoCollection {len(self)} species {hex(id(self))}>"
        return f"<XenoCantoCollection {hex(id(self))}>"
    
    #------------------------------------
    # __call__ 
    #-------------------
    
    def __call__(self, one_per_bird_phylo=True):
        '''
        This is funky; sorry! Defining this method makes
        a XenoCantoCollection instance callable, in this
        case with a keyword arg. This enables the equivalent
        to being able to pass an argument to iter():
        
            class SomeClass:
                def __init__(self):
                    self.i = 0
                def __iter__(self):
                    return self
                def __next__(self):
                    self.i += 1
                    if self.i > 5:
                        self.one_per_bird_phylo = False
                        self.i = 0
                        raise StopIteration
                    return self.i
                def __call__(self, one_per_bird_phylo=False):
                    self.one_per_bird_phylo=one_per_bird_phylo
                    return self
            
            inst = SomeClass()
            for rec in inst(one_per_bird_phylo=True):    # <-----------
                print (inst.i, inst.one_per_bird_phylo)

        Without this trick, iterators aren't callable. 

        :param one_per_bird_phylo:
        :type one_per_bird_phylo:
        '''
        self.one_per_bird_phylo=one_per_bird_phylo
        return self

    #------------------------------------
    # save 
    #-------------------
    
    def save(self, dest=None):
        '''
        Saves collection to given file. Only the
        metadata for the collection itself, and
        the enclosed XenoCantoRecording instances
        are saved, not the soundfiles themselves.
        They are normally in their own dir, with 
        the XenoCantoRecording instances holding
        the file paths.
        
        If dest:
        
           o is an existing directory, a .json file name 
                is created for the collection
                file.
           o is a file name as ascertained by the
                presence of an extension: 
                If the dest file ends with '.json' 
                the output format is JSON. If it 
                ends with '.pkl' or '.pickle' the 
                sound_collection is saved in pickle format.
                Any other extension is an error.
            o Is None: the destination directory
                will be a subdir 'xeno_canto_collections'
                under this script's dir.
                
        In all cases, non-existing directories
        will be created. 
        
        Note: Pickle is fast to load, but very finicky 
              about file names and directory structures
              being exactly what they were at saving
              time. So JSON is recommended.
        
        :param dest: destination directory or
            file path for the saved collection. 
            Default: 
                <dir_of_this_script>/xeno_canto_collections
        :type dest: str
        :return the full path of the output file
        :rtype str
        '''

        # To distinguish between dest
        # being a file name vs. at dir:
        is_file = False
        
        if dest is None:
            curr_dir  = os.path.dirname(__file__)
            dest_dir  = os.path.join(curr_dir, 'xeno_canto_collections')

        else:
            # Is dest a file or a dir?
            dest_p = Path(dest)
            suffix = dest_p.suffix
            if len(suffix) == 0:
                dest_dir = dest
            else:
                # Destination is a file name:
                is_file = True
                # Check for legal extension:
                if suffix not in ('.json', '.pkl', '.pickle'):
                    raise ValueError("Collection files must end in .json, .pkl, or .pickle")
                dest_dir = dest_p.parent

        if not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir)
            except FileExistsError:
                pass

        # If dest is a directory, invent a filename:
        if not is_file:
            # Create a non-existing .JSON file name 
            # in dest_dir
            dest = self.create_filename(dest_dir) 
        
        # Dest is a file at this point.
        dest_p     = Path(dest)
        dir_part   = dest_p.parent
        file_part  = dest_p.stem
        extension  = dest_p.suffix
        uniquifier = 0
        orig_file_part = file_part
        
        while dest_p.exists():
            #answer = input(f"File {dest_p.name} exists; overwrite? (y/N): ")
            #if answer not in ('y','Y','yes','Yes', ''):
            #    ...
            # Keep adding numbers to end of file
            # till have a unique name. First, remove
            # a unifier we may have added in a prev.
            # run through this loop:
            if file_part.endswith(f"_{uniquifier}"):
                file_part = orig_file_part 
            uniquifier += 1
            file_part += f"_{uniquifier}"
            dest_p = Path.joinpath(dir_part, file_part+extension)

        dest = str(dest_p)

        # At this point dest is a non-existing file,
        # which is what we need. Use pickle or json?
        if extension in ('.json', '.JSON'):
            self.to_json(dest, force=True)
        else:
            raise DeprecationWarning("Deprecated! Using pickle for saving is unreliable")
            with open(dest, 'wb') as fd:
                pickle.dump(self, fd)

        return dest

    #------------------------------------
    # load
    #-------------------
    
    @classmethod
    def load(cls, src):
        '''
        Takes path to either a .json or 
        a .pkl (or .pickle) file. Materializes
        the XenoCantoCollection that is
        encoded in the file.

        :param src: full path to pickle or json file
            of previously saved collection
        :type src: str
        '''

        if not os.path.exists(src) or not(os.path.isfile(src)):
            raise FileNotFoundError(f"Recording collection file {src} does not exist")

        if Path(src).suffix not in ('.json', '.JSON', '.pkl', '.pickle'):
            raise ValueError(f"Saved collection must be a JSON or pickle file")

        # Allow use of tilde in fnames:
        src = os.path.expanduser(src)
        if src.endswith('.json'):
            return cls.from_json(src=src)
        else:
            # Assume pickle file
            with open(src, 'rb') as fd:
                return pickle.load(fd, None)

    #------------------------------------
    # to_json 
    #-------------------
    
    def to_json(self, dest=None, force=False):
        '''
        Creates a JSON string from this 
        collection. If dest is a string,
        it is assumed to be a destination
        file where the json will be saved.
        If none, the generated JSON string 
        is returned. It can be passed to 
        from_json() in the json_str
        kwarg.
        
        If the file exists, the user is warned,
        unless force is True.

        :param dest: optional destination file
        :type dest: {None | str}
        :param force: set to True if OK to overwrite
            dest file. Default: ask permission
        :type force: bool
        :return destination file name if written to
            file, else the JSON string
        '''

        # self.values() are lists of XenoCantoRecording
        # instances. The default specification 
        # tells orjson whom to call with one
        # of those instances to get a json snippet.
        
        jstr = orjson.dumps(self.data,
                            default=XenoCantoRecording._mk_json_serializable)

        if dest is None:
            return jstr
        
        if os.path.exists(dest) and not force:
            answer = input(f"File {os.path.basename(dest)} exists; overwrite? (y/N): ")
            if answer not in ('y','Y','yes','Yes', ''):
                self.log.info("Collection JSON save aborted on request")
                return None

        # At this point dest is a non-existing file,
        # or one that exists but ok to overwrite:
        with open(dest, 'wb') as fd:
            if type(jstr) != bytes:
                jstr = str.encode(jstr)
            fd.write(jstr)
            
        return dest

    #------------------------------------
    # from_json 
    #-------------------
    
    @classmethod
    def from_json(cls, src):
        '''
        Load a collection either from a JSON string,
        or from a file that contains JSON.
        
        :param src: either a json string to parse,
            or the path to a file ending with
            either .json or .JSON
        :type src: str
        :return: the loaded collection
        :rtype: XenoCantoCollection
        '''
        if Path(src).suffix in ('.json', '.JSON'):
            # Read JSON from file:
            if not os.path.exists(src):
                raise FileNotFoundError(f"File {src} not found")
            with open(src, 'rb') as fd:
                json_str = fd.read()
        else:
            # src is assumed to be a json string
            json_str = src
            
        inst_vars = orjson.loads(json_str)
        
        # We now have a dict (the collection) whose
        # values are lists of dicts. Each of these
        # dicts is json code for one XenoCantoRecording.
        
        # First, get an empty XenoCantoCollection,
        # but with care: 
        # This method (from_json()) may be
        # called from __new__(). That following
        # call will call __new__() recursively.
        # But: passing None will make the __new__()
        # method aware, and it will just create
        # an empty instance, for which __init__()
        # will have been called:
        inst = XenoCantoCollection(None)

        for phylo_nm, rec_dict_list in inst_vars.items():
            recs = [XenoCantoRecording.from_json(rec_dict)
                    for rec_dict
                    in rec_dict_list
                    ]
            inst[phylo_nm] = recs
            
        return inst

    #------------------------------------
    # create_filename
    #-------------------
    
    def create_filename(self, dest_dir, extension='.json'):
        '''
        Create a file name that is not in the 
        given directory.
        
        :param dest_dir:
        :type dest_dir:
        :return filename 
        :rtype str
        '''
        t = datetime.datetime.now()
        orig_filename_root = t.isoformat().replace('-','_').replace(':','_')
        uniquifier = 0
        filename = orig_filename_root
        full_path = os.path.join(dest_dir,filename)
        while os.path.exists(full_path):
            uniquifier += 1
            filename = os.path.join(f"{orig_filename_root}_{str(uniquifier)}",
                                    extension
                                    )
            full_path = os.path.join(dest_dir,filename)
        full_path += extension
        return full_path

# --------------------------- Class XenoCantoRecording

class XenoCantoRecording:


    # No decision made about whether
    # to overwrite existing recording files
    # when asked to download any of the
    # instances. Will be updated or used
    # during calls to download():

    always_overwrite = None
    
    # Directory to use for download destination dir
    # if a dir specified in the call to download()
    # is protected (i.e. permissions). Once the user
    # was asked for a replacement dir, the following
    # class var will be set, and used throughout the 
    # lifetime of this instance:
    
    default_dest_dir = None 
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 recording_metadata, 
                 dest_dir=None, 
                 log=None
                 ):
        '''
        Recording metadata must be a dict from the 
        'recordings' entry of a Xeno Canto download.
        The dict contains much info, such as bird name, 
        recording_metadata location, length, sample rate,
        4-letter and 6-letter codes, and file download 
        information. Create instance vars from just 
        some of them.

        :param recording_metadata: a 'recordings' entry from a
            XenoCanto metadata download of available recordings
        :type recording_metadata: {str | {str : Any}|
        :param dest_dir: directory were to store downloaded
            sound files. If None, creates subdirectory of
            this script called: 'recordings'
        :type dest_dir: {None | str}
        :param log: logging service; if None, creates one
        :type log: {None | LoggingService}
        '''
        
        if log is None:
            self.log = LoggingService()
        else:
            self.log = log
            
        # 'Secret' entry: used when creating
        # from json string (see from_json()):
        
        if recording_metadata is None:
            # Have caller initialize the instance vars
            # Used when creating an instance
            # from JSON.
            return
         
        curr_dir = os.path.dirname(__file__)
        if dest_dir is None:
            self.dest_dir = os.path.join(curr_dir, 'recordings')
            if not os.path.exists(self.dest_dir):
                os.mkdir(self.dest_dir)
        else:
            self.dest_dir = dest_dir

        self._xeno_canto_id = recording_metadata['id']
        self.genus     = recording_metadata['gen']
        self.species   = recording_metadata['sp']
        self.four_code = recording_metadata['four_code']
        self.six_code = recording_metadata['six_code']
        self.phylo_name= f"{self.genus}{self.species}"
        self.full_name = f"{self.phylo_name}_xc{self._xeno_canto_id}"
        
        self.country = recording_metadata['cnt']
        self.loc = recording_metadata['loc']
        self.recording_date = recording_metadata['date']
        self.length = recording_metadata['length']
        # One of A-E, or 'no score'
        self.rating = recording_metadata['q']

        # '.mp', '.wav':
        self.encoding = Path(recording_metadata['file-name']).suffix
        
        # Whether 'call' or 'song'
        self.type = recording_metadata['type']
        
        # Like: '//www.xeno-canto.org/482431'
        self._url  = f"HTTP:{recording_metadata['url']}/download"
        
        # Like: 'XC482431-R024 white ruffed manakin.mp3'
        self._filename = recording_metadata['file-name']

    #------------------------------------
    # download 
    #-------------------
    
    def download(self, 
                 dest_root=None, 
                 overwrite_existing=None,
                 testing=False):
        
        if dest_root is None:
            dest_root = self.dest_dir
            
        if overwrite_existing is None:
            # Use the global default. False, unless changed
            # during __init__()
            overwrite_existing = self.always_overwrite

        # If necessary, create a subdir named by the 4-letter
        # code of the bird species:
        try:
            dest_dir = os.path.join(dest_root, self.four_code)
        except TypeError as e:
            # Four-letter code was unavailable in the recording metadata:
            os.path.join(dest_root, f"unknown4letter_xc_{self._xeno_canto_id}")

        while not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir)
            except (PermissionError, OSError):
                
                # Maybe we asked user for a replacement
                # dir for an earlier recording. If so,
                # don't ask again, use their prior answer:
                
                if XenoCantoRecording.default_dest_dir is not None and \
                    os.path.exists(XenoCantoRecording.default_dest_dir):
                    dest_dir = XenoCantoRecording.default_dest_dir
                    continue
                
                # Make a short dir with ellipses if
                # dest_dir very long:
                short_dir = FileUtils.ellipsed_file_path(dest_dir)
                dest_dir = input(f"No permission for {short_dir}; enter new dest folder(tilde OK): ")
                # Resolve '~' notation:
                dest_dir = os.path.expanduser(dest_dir)
                XenoCantoRecording.default_dest_dir = dest_dir
        else:
            if not os.path.isdir(dest_dir):
                raise ValueError(f"Destination {dest_dir} is not a directory")

        # Update the full path to the recording
        # file: if a saved recording is loaded
        # by a different user or on a different
        # machine: that path might not exist:
        
        if not self._has_vocalization_prefix(self._filename):
            self._filename = self._ensure_call_or_song_prefix(
                self._filename, 
                self.type)

        self._filename = self._clean_filename(self._filename)
        self.full_name = os.path.join(dest_dir, self._filename)
        
        # Just to make log info msgs not exceed a terminal
        # line: create an fname with ellipses: '/foo/.../bar/fum.txt'
         
        fname_descr = FileUtils.ellipsed_file_path(self.full_name)

        # Dest directory exists, does the sound file
        # already exist?
        if os.path.exists(self.full_name):
            if overwrite_existing:
                go_ahead = True
            else:
                # Need to ask permission to overwrite:
                answer = input(f"Recording {self._filename} exists; overwrite (y/N): ")
                go_ahead = answer in ('y','Y','yes','Yes')
                # Make answer the global default for recordings:
                self.__class__.always_overwrite = go_ahead
        else:
            go_ahead = True
            
        if testing:
            return go_ahead
        
        if not go_ahead:
            self.log.info(f"Skipping {self._filename}: already downloaded.")
            return self.full_name
        
        self.log.info(f"Downloading {self.four_code}: {fname_descr}...")
        try:
            response = requests.get(self._url)
        except Exception as e:
            raise IOError(f"While downloading {self._url}: {repr(e)}") from e
        
        self.log.info(f"Done downloading {fname_descr}")
        
        # Split "audio/mpeg" or "audio/vdn.wav"
        medium, self.encoding = response.headers['content-type'].split('/')
        
        if medium != 'audio' or self.encoding not in ('mpeg', 'vdn.wav'):
            msg = f"Recording {self.full_name} is {medium}/{self.encoding}, not mpeg or vpn_wav"
            self.log.err(msg)
            #raise ValueError(msg)
        else:
            self.log.info(f"File encoding: {self.encoding}")
        
        # Add 'CALL' or 'SONG' in front of filename
        self._filename = self._ensure_call_or_song_prefix(
            self._filename, self.type)
        
        self.file_name = os.path.join(dest_dir, self._filename)
        
        self.log.info(f"Saving {fname_descr}...")
        with open(self.full_name, 'wb') as f:
            f.write(response.content)
            self.log.info(f"Done saving {fname_descr}")
        return self.full_name

    #------------------------------------
    # _clean_filename
    #-------------------
    
    def _clean_filename(self, fname):
        '''
        People put the horriblest chars into 
        filenames: spaces, parentheses, backslashes!
        Replace any of those with underscores.
        
        :param fname: original name
        :type fname: str
        :return: cleaned up, unix-safe name
        :rtype: str
        '''
        fname = fname.replace('/', '_')
        fname = fname.replace(' ', '_')
        fname = fname.replace('(', '_')
        fname = fname.replace(')', '_')
        
        return fname

    #------------------------------------
    # _ensure_call_or_song_prefix
    #-------------------
    
    def _ensure_call_or_song_prefix(self, path, vocalization_type):
        '''
        Given a path (just a file name or a 
        path in a subdir), prefix the file name
        with CALL_ or SONG_, depending on the 
        vocalization_type. The vocalization_type can
        actually be anything else to use as prefix.
        
        :param path: current path without the CALL or SONG
        :type path: str
        :param vocalization_type: usually 'CALL' or 'SONG'
        :type vocalization_type: str
        :return: new path with the prefix in place
        :rtype: str
        '''
        
        # Already has the prefix?
        if self._has_vocalization_prefix(path):
            return path
        p = Path(path)
        
        # Sometimes people put things like:
        # 'ADULT, SEX UNCERTAIN, SONG_XC5311'... or
        # 'ALARM CALL_20110913FORANAal.mp3' or
        # 'CALL, FEMALE, MALE_XC441963-FormicanalVZA38a.mp3'
        # Into the type field. See whether we
        # find 'CALL' or 'SONG'. If not, prefix
        # with UNKNOWN:
        
        if vocalization_type not in ('CALL', 'SONG', 'call', 'song'):
            voctype_lowcase = vocalization_type.lower()
            if voctype_lowcase.find('call') > -1:
                vocalization_type = 'CALL'
            elif voctype_lowcase.find('song') > -1:
                vocalization_type = 'SONG'
            else:
                vocalization_type = 'UNKNOWN'
         
        fname = f"{vocalization_type.upper()}_{p.name}"
        new_path = str(p.parent.joinpath(fname))
        return new_path

    #------------------------------------
    # _has_vocalization_prefix 
    #-------------------
    
    def _has_vocalization_prefix(self, path):
        
        fname = os.path.basename(path)
        return fname.startswith('CALL') or fname.startswith('SONG') 

    #------------------------------------
    # __repr__ 
    #-------------------
    
    def __repr__(self):
        return f"<XenoCantoRecording {self.full_name} {hex(id(self))}>"

    #------------------------------------
    # __str__ 
    #-------------------
    
    def __str__(self):
        return self.__repr__()

    #------------------------------------
    # __eq__
    #-------------------
    
    def __eq__(self, other_recording):
        for inst_var_nm, inst_var_val in self.__dict__.items():
            if type(inst_var_val) == str:
                if other_recording.__dict__[inst_var_nm] != inst_var_val:
                    return False
            elif type(inst_var_val) == LoggingService:
                if type(other_recording.__dict__[inst_var_nm]) != LoggingService:
                    return False
            else:
                # Inst var of unexpected type:
                return False
        return True

    #------------------------------------
    # to_json 
    #-------------------
    
    def to_json(self):
        
        return orjson.dumps(self, 
                            default=XenoCantoRecording._mk_json_serializable)

    #------------------------------------
    # _mk_json_serializable
    #-------------------
    
    def _mk_json_serializable(self):
        '''
        Called by orjson to json-serialize
        a XenoCantoRecording instance. The method
        extracts those instance variables of self
        that are strings, and returns the resulting
        dict. 
        '''
        as_dict = {inst_var_nm : inst_var_value
                   for inst_var_nm, inst_var_value
                   in self.__dict__.items()
                   if type(inst_var_value) == str
                   }
        return as_dict

    #------------------------------------
    # from_json 
    #-------------------
    
    @classmethod
    def from_json(cls, json_str_or_dict):
        if type(json_str_or_dict) in (str, bytes):
            # Get a dict if one wasn't passed in:
            inst_vars = orjson.loads(json_str_or_dict)
        else:
            inst_vars = json_str_or_dict
            
        inst = cls.from_dict(inst_vars)
        return inst 

    #------------------------------------
    # from_dict 
    #-------------------
    
    @classmethod
    def from_dict(cls, inst_var_dict):
        inst = XenoCantoRecording(None)
        try:
            inst.__dict__.update(inst_var_dict)
        except TypeError as e:
            print(f"Err: {repr(e)}")
        return inst

# ------------------------ Main ------------
if __name__ == '__main__':
    
#     birds = ['Tangara+gyrola', 'Amazilia+decora', 'Hylophilus+decurtatus', 'Arremon+aurantiirostris', 
#              'Dysithamnus+mentalis', 'Lophotriccus+pileatus', 'Euphonia+imitans', 'Tangara+icterocephala', 
#              'Catharus+ustulatus', 'Parula+pitiayumi', 'Henicorhina+leucosticta', 'Corapipo+altera', 
#              'Empidonax+flaviventris']

    examples = '''
    Examples:
        xeno_canto_manager.py --download "Tangara gyrola"
        xeno_canto_manager.py --download BHTA "Lophotriccus pileatus"
        
        xeno_canto_manager.py \\
                 --destdir /home/data/birds/Soundfiles/XCForAllLabeledRecs \\
                 --download \\
                 --overwrite \\
                 --all_recordings \\
                 LEMO BGDO YTTO COTF GHOT GKIS HOWO MITY RBPE RCRW SHWO        
    '''

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Download and process Xeno Canto bird sounds",
                                     epilog=examples
                                     )
  
    parser.add_argument('-d', '--destdir',
                        help='fully qualified directory for downloads. Default: /tmp',
                        default='/tmp')
    parser.add_argument('-t', '--timedelay',
                        type=float,
                        help='time between downloads for politeness to XenoCanto server. Ex: 1.0',
                        default=1.0)
    parser.add_argument('-c', '--collection',
                        type=str,
                        help='optionally path to existing, previously saved collection',
                        default=None)
    # Things user wants to do:
    parser.add_argument('--collect_info',
                        action='store_true',
                        help="download metadata for the bird calls, but don't download sounds",
                        default=True
                        )
    parser.add_argument('--download',
                        action='store_true',
                        help="download the (birds_to_process) sound files (implies --collect_info",
                        default=False
                         
                        )
    parser.add_argument('--all_recordings',
                        action='store_true',
                        help="download all recordings rather than one per species; default: one per.",
                        default=False
                        )
    parser.add_argument('--overwrite',
                        action='store_true',
                        help="whether or not to overwrite already downloaded files w/o asking; Def: False",
                        default=False
                        )
    parser.add_argument('birds_to_process',
                        type=str,
                        nargs='+',
                        help='Repeatable: <genus>_<species> or "<genus> <species>". Ex: Tangara_gyrola')
  
    args = parser.parse_args()
  
    if args.collection:
        sound_collection = XenoCantoCollection.load(args.collection)
    else:
        sound_collection = None
  
    # For reporting to user: list
    # of actions to do by getting the actions
    # as strings from the args instance:
    
    todo = [action_name 
            for action_name
            in ['collect_info', 'download']
            if args.__getattribute__(action_name)
            ]
      
    if len(todo) == 0:
        print("No action specified on command line; nothing done")
        print(parser.print_help())
        sys.exit(0)
              
    # Fix bird names to make HTTP happy later.
    # B/c we allow -b and --bird in args, 
    # argparse constructs a name:
      
    birds_to_process = args.birds_to_process

    # Turn 4-letter or 6-letter codes into
    # their scientific names:
    
    birds_scientific = []
    bird_code_cnv = SpeciesNameConverter()
    for bird in birds_to_process:
        try:
            if len(bird) == 4:
                birds_scientific.append(bird_code_cnv[bird,
                                                      DIRECTION.FOUR_SCI 
                                                      ])
            elif len(bird) == 6:
                birds_scientific.append(bird_code_cnv[bird,
                                                      DIRECTION.SIX_SCI
                                                      ])
        except KeyError:
            # Could not find scienfic name of a
            # 4-letter or 6-letter species. Since
            # we have not started yet, stop now:
            print(f"Could not convert {bird} into scientific name")
            sys.exit(-1)
            
        else:
            # Bird is already a scientific name.
            # Replace underscores needed for
            # not confusing bash with the '+' signs
            # that are required in URLs:
            birds_scientific.append(bird.replace('_', '+'))
            birds_scientific.append(bird.replace(' ', '+'))
          
    if sound_collection is None and \
        (('collect_info' in  todo) or\
         ('download' in todo)
        ):
        # Request to download recordings or
        # metadata, and no already existing and
        # saved collection was specified: 
        
        sound_collection = XenoCantoCollection(birds_scientific,
                                               dest_dir=args.destdir,
                                               always_overwrite=args.overwrite
                                               )
        saved_path = sound_collection.save()
        sound_collection.log.info(f"Saved new collection to {saved_path}")
          
    if 'download' in todo:
        one_per_species = not args.all_recordings
        sound_collection.download(birds_scientific,
                                  one_per_species=one_per_species,
                                  overwrite_existing=args.overwrite,
                                  courtesy_delay=args.timedelay)
        # Save the updated collection:
        saved_path = sound_collection.save(saved_path)
        sound_collection.log.info(f"Saved updated collection to new files: {saved_path}")
  
    sound_collection.log.info(f"Done with {todo}")

# ------------------------ Testing Only ----------------
    # Testing (Should move to a unittest
#     sound_collection = XenoCantoCollection(['Tangara+gyrola', 
#                                             'Amazilia+decora'],
#                                             dest_dir='/tmp'
#                                             )
#     #rec = next(iter(sound_collection, one_per_bird_phylo=False))
#     for rec in sound_collection(one_per_bird_phylo=False):
#         print(rec.full_name)
#     for rec in sound_collection(one_per_bird_phylo=True):
#         print(rec.full_name)
#         
#     rec.download()
#     print(sound_collection)
#     
#     sound_collection = XenoCantoCollection.from_json(src='/Users/paepcke/EclipseWorkspacesNew/birds/test_metadata.json')
#     for rec in sound_collection:
#         rec.download()
#         print(rec)
#     
#     new = XenoCantoCollection.load('/Users/paepcke/tmp/test_metadata1.pkl')
#     sound_collection = XenoCantoCollection.load('/Users/paepcke/tmp/test_metadata1.json')
#     for rec in sound_collection:
#         print(rec)
#     
#     for rec in sound_collection:
#         rec.download()
#     
#     print(sound_collection)
