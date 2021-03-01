#!/usr/bin/env python3
"""
Functions involved in downloading birds from xeno-canto
"""

import argparse
from collections import UserDict
import os
from pathlib import Path
import re
import sys
import time

import librosa
import librosa.display
from logging_service import LoggingService
import requests
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np


class XenoCantoProcessor:
    
    def __init__(self, load_dir=None, bird_names=None):
    
            
        if bird_names is not None:
            self.recording_collection = self.xeno_canto_get_bird_metadata(bird_names)
            #print(sounds)
            
    def process_recording(self, birdname):
        #read back from file
        filepath = './Birdsong/' + birdname + '.wav'
        
        audiotest, sr = librosa.load(filepath, sr = None)      #fix this
        dur = librosa.get_duration(audiotest, sr)
        
        for i in range (0, int(dur/5)):
            audio, sr = librosa.load(filepath, offset = 5.0 * i, duration = 5.0, sr = None)
            #filter the bird audio
            audio = self.filter_bird(birdname, str(i), audio, sr)
        
            #create and save spectrogram
            self.create_spectrogram(birdname, str(i), audio, sr)
        
        
    def filter_bird(self, birdname, instance, audio, sr):
        #bandpass
        b, a = self.define_bandpass(2000, 3500, sr)
        output = signal.lfilter(b, a, audio)
        
        #output = output - np.mean(output)
        
        #write filtered file
        outfile = './Birdsong_Filtered/' + birdname + instance + '.wav' 
        librosa.output.write_wav(outfile, output, sr)
        return output
        
    #definition and implementation of the bandpass filter
    def define_bandpass(self, lowcut, highcut, sr, order = 2):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype = 'band')
        return b, a
        
    def create_spectrogram(self, birdname, instance, audio, sr, n_mels = 128):
        spectrogramfile = './Birdsong_Spectrograms/' + birdname + instance + '.jpg' 
        mel = librosa.feature.melspectrogram(audio, sr = sr, n_mels = n_mels)
        log_mel = librosa.power_to_db(mel, ref = np.max)
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.savefig(spectrogramfile, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close()   

# ---------------- Class XenoCantoCollection ----------

class XenoCantoCollection(UserDict):
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

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, 
                 bird_names,
                 courtesy_delay=1.0,
                 load_dir=None):

        super().__init__()
        
        self.log = LoggingService()
        self.courtesy_delay = courtesy_delay
        
        curr_dir = os.path.dirname(__file__)
        if load_dir is None:
            self.load_dir = os.path.join(curr_dir, 'recordings')
            if not os.path.exists(self.load_dir):
                os.mkdir(self.load_dir)
        else:
            self.load_dir = load_dir
            
        # Default behavior of iterator (see comment in __iter__):
        self.one_per_bird_phylo=True
    
        collections_metadata = self.xeno_canto_get_bird_metadata(bird_names)
        for species_recs_metadata in collections_metadata:
            # Metadata about XC holdings for one genus/species:
            self.num_recordings = species_recs_metadata['numRecordings']
            self.num_species = species_recs_metadata['numSpecies']
            # List of dicts, each describing one 
            # recording:
            recordings_list  = species_recs_metadata['recordings']
    
            # Create XenoCantoRecording instances
            # without sound file download. That is
            # done lazily:
            
            for rec_dict in recordings_list:
                rec_obj  = XenoCantoRecording(rec_dict, 
                                              load_dir=self.load_dir, 
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
        metadata = []
        for bird_name in bird_names:
            url = "https://www.xeno-canto.org/api/2/recordings"
            http_query = f"{url}?query={bird_name}+len:5-60+q_gt:C"
            
            self.log.info(f"Downloading metadata for {len(bird_names)} bird(s)...")
            try:
                response = requests.get(http_query)
            except Exception as e:
                self.log.err(f"Failed to download metadata for {bird_name}: {repr(e)}")
                time.sleep(self.courtesy_delay)
                continue
            self.log.info(f"Done downloading metadata for {len(bird_names)} bird(s)...")
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
        
        # For convenienct:
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
        phylo_name, track phylo_names with curr_bird_phylo_indx
        into the list of phylo_name_n. The 'next' recording is
        the first recording in the next phylo_name.
        
        If iterator is to feed every bird, two indices are needed
        the one into phylo_names, and one into the current phylo_name
        value's list of recordings: self.curr_recording_indx. For
        each phylo_name we feed one recording after the other,  until
        the list is exhausted, then move on to the next phylo_name.
        
        @return: Xeno Canto recording instance
        @rtype: XenoCantoRecording
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
    # download 
    #-------------------
    
    def download(self, 
                 birds_to_process=None,
                 one_per_species=True, 
                 courtesy_delay=1.0):
        '''
        Download a given list of species, or
        all the species in the collection. 
        
        If one_per_species is True, only one recording
        from the list of each species' recordings is
        downloaded; else all recordings of each requested
        species.
        
        Courtesy delay is time to wait between sound file
        downloads.
        
        @param birds_to_process: list of bird species names
        @type birds_to_process: [str]
        @param one_per_species: whether or not to download all 
            recordings of each species, or just one
        @type one_per_species: bool
        @param courtesy_delay: time between requests to XC server
        @type courtesy_delay: {int | float}
        '''
        
        if birds_to_process is None:
            birds_to_process = list(self.keys())

        # Get number of sound files to download:
        if one_per_species:
            num_to_download = len(birds_to_process)
        else:
            num_to_download = 0
            for species in birds_to_process:
                try:
                    num_to_download += self[species]
                except KeyError:
                    self.log.err(f"Species {species} not represented in collection")

        downloaded = 1
        for rec in self(one_per_bird_phylo=one_per_species):
            self.log.info(f"{downloaded}/{num_to_download}")
            rec.download()
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

        @param one_per_bird_phylo:
        @type one_per_bird_phylo:
        '''
        self.one_per_bird_phylo=one_per_bird_phylo
        return self


# --------------------------- Class XenoCantoRecording

class XenoCantoRecording:
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, recording_metadata, load_dir=None, log=None):
        '''
        Recording must be a dict from the 
        'recordings' entry of a Xeno Canto download.
        The dict contains much info, such as bird name, 
        recording_metadata location, length, sample rate, and
        file download information. Create instance vars
        from just some of them.

        @param recording_metadata: a 'recordings' entry from a
            XenoCanto metadata download of available recordings
        @type recording_metadata: {str | {str : Any}|
        @param load_dir: directory were to store downloaded
            sound files. If None, creates subdirectory of
            this script called: 'recordings'
        @type load_dir: {None | str}
        @param log: logging service; if None, creates one
        @type log: {None | LoggingService}
        '''
        
        if log is None:
            self.log = LoggingService()
        else:
            self.log = log
        curr_dir = os.path.dirname(__file__)
        if load_dir is None:
            self.load_dir = os.path.join(curr_dir, 'recordings')
            if not os.path.exists(self.load_dir):
                os.mkdir(self.load_dir)
        else:
            self.load_dir = load_dir
        
        self._xeno_canto_id = recording_metadata['id']
        self.genus     = recording_metadata['gen']
        self.species   = recording_metadata['sp']
        self.phylo_name= f"{self.genus}{self.species}"
        self.full_name = f"{self.phylo_name}_xc{self._xeno_canto_id}"
        
        self.country = recording_metadata['cnt']
        self.loc = recording_metadata['loc']
        self.recording_date = recording_metadata['date']
        self.length = recording_metadata['length']

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
    
    def download(self, dest_dir=None):
        if dest_dir is None:
            dest_dir = self.load_dir
        
        self.log.info(f"Downloading sound file for {self.full_name}...")
        response = requests.get(self._url)
        self.log.info(f"Done downloading sound file for {self.full_name}")
        
        # Split "audio/mpeg" or "audio/vdn.wav"
        medium, self.encoding = response.headers['content-type'].split('/')
        
        if medium != 'audio' or self.encoding not in ('mpeg', 'vdn.wav'):
            msg = f"Recording {self.full_name} is {medium}/{self.encoding}, not mpeg or vpn_wav"
            self.log.err(msg)
            #raise ValueError(msg)
        else:
            self.log.info(f"File encoding: {self.encoding}")
            
        ext = 'mp3' if self.encoding == 'mpeg' else 'wav' 
        filename = f"{self.type.upper()}_{self.full_name}.{ext}" 
        self.filepath = os.path.join(dest_dir,filename)
        
        self.log.info(f"Saving sound file to {self.filepath}...")
        with open(self.filepath, 'wb') as f:
            f.write(response.content)
            self.log.info(f"Done saving sound file to {self.filepath}")
        return 

    #------------------------------------
    # read_recording_specs 
    #-------------------
    
    def read_recording_specs(self, filename):
        '''
        If possible, reads the following properties
        from the sound file:
        
            Audio file properties
            File type	mp3
            Length	42.3 (s)
            Sampling rate	48000 (Hz)
            Bitrate of mp3	196496 (bps)
            Channels	2 (stereo)
                    
        @param filename:
        @type filename:
        '''
        
        pass

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

# ------------------------ Main ------------
if __name__ == '__main__':
    
    birds = ['Tangara+gyrola', 'Amazilia+decora', 'Hylophilus+decurtatus', 'Arremon+aurantiirostris', 
             'Dysithamnus+mentalis', 'Lophotriccus+pileatus', 'Euphonia+imitans', 'Tangara+icterocephala', 
             'Catharus+ustulatus', 'Parula+pitiayumi', 'Henicorhina+leucosticta', 'Corapipo+altera', 
             'Empidonax+flaviventris']

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Download and process Xeno Canto bird sounds"
                                     )

    parser.add_argument('-d', '--destdir',
                        help='fully qualified directory for downloads. Default: /tmp',
                        default='/tmp')
    parser.add_argument('-t', '--timedelay',
                        type=float,
                        help='time between downloads for politeness to XenoCanto server. Ex: 1.0',
                        default=1.0)
    # Things user wants to do:
    parser.add_argument('--collect_info',
                        action='store_true',
                        help="download metadata for the bird calls, but don't download sounds"
                        )
    parser.add_argument('--download',
                        action='store_true',
                        help="download the sound files (implies --collect_info"
                        )
    parser.add_argument('--one_each_species',
                        action='store_true',
                        help="only download one recording for each bird; default: True",
                        default=True
                        )
    parser.add_argument('birds_to_process',
                        type=str,
                        nargs='*',
                        help='Repeatable: <genus>+<species>. Ex: Tangara_gyrola',
                        default=[bird.replace('+', '_') for bird in birds])

    args = parser.parse_args()

    # For reporting to user: list
    # of actions to do:
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
    
    if len(birds_to_process) == 0:
        birds_to_process = birds
    else:
        # Replace the underscores needed for
        # not confusing bash with the '+' signs
        # required in URLs:
        birds_to_process = [bird.replace('_', '+') for bird in birds_to_process]

    #if len(birds_to_process) == 0:
    #    print("No birds specified; nothing done")
    #    sys.exit(0)
        
    if ('collect_info' in  todo) or ('download' in todo):
        sound_collection = XenoCantoCollection(birds_to_process,
                                               load_dir=args.destdir)
        
    if 'download' in todo:
        sound_collection.download(birds_to_process,
                                  one_per_species=args.one_per_species, 
                                  courtesy_delay=args.timedelay)
        
    sound_collection.log.info(f"Done with {todo}")

# ------------------------ Testing Only ----------------
    # Testing
#     sound_collection = XenoCantoCollection(['Tangara+gyrola', 
#                                             'Amazilia+decora'],
#                                             load_dir='/tmp'
#                                             )
#     #rec = next(iter(sound_collection, one_per_bird_phylo=False))
#     for rec in sound_collection(one_per_bird_phylo=False):
#         print(rec.full_name)
#     for rec in sound_collection(one_per_bird_phylo=True):
#         print(rec.full_name)
#         
#     rec.download()
#     print(sound_collection)
