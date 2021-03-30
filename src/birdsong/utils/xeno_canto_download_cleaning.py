'''
Created on Mar 28, 2021

@author: paepcke
'''
import os
import shutil

from birdsong.xeno_canto_manager import XenoCantoCollection

class BirdCorrector(object):
    '''
    A few methods used to rename and otherwise
    cleanup a large Xeno Canto download. Hopefully
    this file won't be needed in future downloads. 
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self):
        '''
        Constructor
        '''
        
    #------------------------------------
    # make_species_subdirs
    #-------------------
    
    def make_species_subdirs(self, coll_loc, download_dir):
        '''
        Create subdirectories below the download_dir, one
        for each species. Move recordings into the proper
        subdirs.

        @param coll_loc: path to previously saved XenoCantoCollection
        @type coll_loc: str
        @param download_dir: path to where the sound files are located
        @type download_dir: str
        '''
        coll = XenoCantoCollection(coll_loc)
        #*************
        all_recs = coll.all_recordings()
#         for rec in all_recs:
#             rec._filename = self._clean_filename(rec._filename)
#             type_upper = rec.type.upper() 
#             if type_upper in ['SONG', 'CALL']:
#                 curr_fn = rec._filename
#                 if curr_fn.startswith(type_upper):
#                     continue
#                 else:
#                     rec._filename = f"{type_upper}_{rec._filename}"
#         coll.save(coll_loc)
#         return
        #*************
        
        for species in coll.keys():
            try:
                os.mkdir(os.path.join(download_dir, species))
            except FileExistsError:
                # No problem
                pass
            
            # Move all sound files of this species
            # into the proper subdirectory:
            for rec in coll[species]:
                species_dir = os.path.join(download_dir,
                                           species
                                           ) 
                voc_type = rec.type.upper()
                    
                if voc_type.find('CALL') > -1:
                    voc_type = 'CALL'
                    rec.type = 'CALL'
                elif voc_type.find('SONG') > -1:
                    voc_type = 'SONG'
                    rec.type = 'SONG'
                    
                fname = rec._filename
                if voc_type in ('CALL', 'SONG'):
                    if not (fname.startswith('CALL_') or \
                            fname.startswith('SONG_')):
                        # If starts with UNKNOWN_, take
                        # that away first:
                        if fname.startswith('UNKNOWN_'):
                            fname = fname[len('UNKNOWN_'):]
                        fname = f"{voc_type}_{fname}"
                else:
                    fname = f"UNKNOWN_{rec._filename}"

                curr_path = os.path.join(download_dir,
                                        fname 
                                        )
                
                if not os.path.exists(curr_path):
                    print(f"Did not find {curr_path}")
                    continue
                # Update the record's _filename:
                # Replace crud from file name:
                clean_fname = self._clean_filename(fname)
                rec._filename = clean_fname
                dest = os.path.join(species_dir, clean_fname)
                shutil.move(curr_path, dest)
                
        # Save updated collection.
        # The save() method will not
        # overwrite, but create a new
        # json file:
        
        new_json_path = coll.save(coll_loc)
        print(f"Updated collection: {new_json_path}")

    #------------------------------------
    # unknown_type_resolution
    #-------------------

    def unknown_type_resolution(self, coll_loc, download_dir):
        '''
        Downloaded recordings often have non-standard
        type fields, making the distinction between song
        and call difficult. The downloaded sound files then
        end up with UNKNOWN_ prefixes. 
        
        This method loads the saved collection that contains
        the metadata for the downloaded files. It tries to 
        guess in each XenoCantoRecording instance the proper 
        type from the type fields. It corrects the type field
        if successful. The UNKNOWN_ prefixed file is then 
        moved to have the proper SONG_/CALL_ prefix.
        
        Prints remaining unresolved cases. 
        
        @param coll_loc: path to previously saved XenoCantoCollection
        @type coll_loc: str
        @param download_dir: path to where the sound files are located
        @type download_dir: str
        '''
        
        #coll_loc = '/Users/paepcke/EclipseWorkspacesNew/birds/src/birdsong/xeno_canto_collections/2021_03_26T18_07_39.560878.json'
        coll = XenoCantoCollection(coll_loc)
        #download_dir = '/Users/paepcke/Project/Wildlife/Birds/CostaRica/Data/DownloadedBirds/'
        all_recs = {}
        for rec_list in coll.values():
            for rec in rec_list:
                corr_fname = self._clean_filename(rec._filename)
                rec._filename = corr_fname
                all_recs['UNKNOWN_' + corr_fname] = rec

        self.all_recs = all_recs
        for fname in os.listdir(download_dir):
            try:
                if not fname.startswith('UNKNOWN_'):
                    continue 
                rec = all_recs[fname]
            except KeyError:
                print(f"Not fnd: {fname}")
                continue
            voc_type = rec.type.upper()
            if voc_type.find('CALL') > -1:
                voc_type = 'CALL'
            elif voc_type.find('SONG') > -1:
                voc_type = 'SONG'
            if voc_type in ['SONG', 'CALL']:
                # Standardize the type field:
                rec.type = voc_type
                new_fname = fname.replace('UNKNOWN_', 
                                          f"{voc_type}_")
                orig = os.path.join(download_dir, fname)
                new  = os.path.join(download_dir, new_fname)
                shutil.move(orig,new)
            # Still unknown:
            print(f"Unknown type: {voc_type}")


        # The save will automatically add
        # a uniquifier at the end of the json
        # file name:
        new_coll_loc = coll.save(coll_loc)
        print(f"Updated collection saved to {new_coll_loc}")

    #------------------------------------
    # _clean_filename
    #-------------------
    
    def _clean_filename(self, fname):
        '''
        People put the horriblest chars into 
        filenames: spaces, parentheses, backslashes!
        Replace any of those with underscores.
        
        @param fname: original name
        @type fname: str
        @return: cleaned up, unix-safe name
        @rtype: str
        '''
        fname = fname.replace('/', '_')
        fname = fname.replace(' ', '_')
        fname = fname.replace('(', '_')
        fname = fname.replace(')', '_')
        
        return fname

# ------------------------ Main ------------
if __name__ == '__main__':
    corr = BirdCorrector()
    
    download_dir = '/Users/paepcke/Project/Wildlife/Birds/CostaRica/Data/DownloadedBirds/'
    #coll_loc     = '/Users/paepcke/EclipseWorkspacesNew/birds/src/birdsong/xeno_canto_collections/2021_03_26T18_07_39.560878.json'
    coll_loc     = '/Users/paepcke/EclipseWorkspacesNew/birds/src/birdsong/xeno_canto_collections/2021_03_28T14_50_39.560878_1_3.json'
    #coll_loc     = '/Users/paepcke/EclipseWorkspacesNew/birds/src/birdsong/xeno_canto_collections/2021_03_26T18_07_39.560878_1_1.json'

    corr.make_species_subdirs(coll_loc, download_dir)
    print('foo')