""" This file downloads bird .wav files from metadata created.
    It will create a folder for each bird species, and automatically separate
    call and song if the species is in the preset birdlist list. Else, it will
    create a single folder for that key in the XenoCantoCollection.
"""
import argparse
from birdsong.xeno_canto_processor import XenoCantoCollection
import os

BIRD_LIST = [('Amazilia+decora', True), ('Arremon+aurantiirostris', True),
             ('Corapipo+altera', True),
             ('Dysithamnus+mentalis', True),
             ('Empidonax+flaviventris', False), ('Euphonia+imitans', False),
             ('Henicorhina+leucosticta', True), ('Hylophilus+decurtatus', False),
             ('Lophotriccus+pileatus', True),
             ('Parula+pitiayumi', True),
             ('Tangara+gyrola', False), ('Tangara+icterocephala', False)
            ]
# Below used for testing
# BIRD_LIST = [('Amazilia+decora', True),
#              # ('Arremon+aurantiirostris', True),
#              # ('Corapipo+altera', True),
#              # ('Dysithamnus+mentalis', True),
#              # ('Empidonax+flaviventris', False), ('Euphonia+imitans', False),
#              # ('Henicorhina+leucosticta', True), ('Hylophilus+decurtatus', False),
#              # ('Lophotriccus+pileatus', True),
#              # ('Parula+pitiayumi', True),
#              ('Tangara+gyrola', False)
#              # ,('Tangara+icterocephala', False)
#             ]


def folder_prefix(birdname):
    name_list = birdname.split('+')
    #Folder prefix is first three letters of first and second part of birdname
    folder_prefix = name_list[0][:3] + name_list[1][:3]
    return folder_prefix.upper()


def download_bird_samples(meta_data_path, out_dir, num_samples_per_species):
    """
    Wrapper function for downloading birds. Iterates through list birds and calls download function for each
    species.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    birds = [pair[0] for pair in BIRD_LIST]
    xc_collection = XenoCantoCollection(birds, load_dir=meta_data_path)
    for birdname, split_call_song in BIRD_LIST:
        # download samples from zeno-canto
        num_records = download_bird(birdname, out_dir, folder_prefix(birdname), split_call_song,
                                    xc_collection[birdname.replace('+', '')],
                                    to_download=num_samples_per_species)
        print(f"Downloaded {num_records} records for {birdname}.")

def download_bird(birdname, out_dir, folder_prefix, split_call_song, xc_recordings, to_download = -1):
    """
    Downloads all recording of a specifc bird species from zeno-canto

    :param bird: the scientific name of the bird to be downloaded
    :type bird: str
    :param out_dir: the relative or absolute file path to a directory to put the output files
    :type out_dir: str
    :returns num: returns the number of records downloaded for the bird
    """
    # Create folders for samples to go in
    # TODO: once merged with takao branch, use utils.make_folder
    if split_call_song:
        # Make folder for call and song
        if not os.path.exists(os.path.join(out_dir, folder_prefix + "_S")):
            os.mkdir(os.path.join(out_dir, folder_prefix + "_S"))
        if not os.path.exists(os.path.join(out_dir, folder_prefix + "_C")):
            os.mkdir(os.path.join(out_dir, folder_prefix + "_C"))
    else:
        # Make one folder for the species
        if not os.path.exists(os.path.join(out_dir, folder_prefix)):
            os.mkdir(os.path.join(out_dir, folder_prefix))

    # response = requests.get("https://www.xeno-canto.org/api/2/recordings?query=" + birdname + "+len:5-60+q_gt:C")
    # data = response.json()

    num_downloaded = 0
    to_download = len(xc_recordings) if to_download == -1 else to_download
    print(f"Metadata for {birdname} returned {len(xc_recordings)} recordings. ")
    for xc_recording in xc_recordings: # each bird may have multiple records
        #download file
        #figure out folder_name
        record_name = xc_recording.full_name
        if split_call_song:
            if isinstance(xc_recording.type, str):
                type_list = xc_recording.type.lower()
            else:
                print('came here')
                type_list = [type.lower() for type in xc_recording.type]
            if 'song' in type_list:
                folder_name = folder_prefix + "_S"
            elif 'call' in type_list:
                folder_name = folder_prefix + "_C"
            else:
                folder_name = folder_prefix
                print(f"Unknown record type: type_list = {type_list}. Skipping {record_name}...")
                continue
            filepath = os.path.join(out_dir, folder_name)
        else:
            filepath = os.path.join(out_dir, folder_prefix)

        # with open(filepath, 'wb') as f:
        #     f.write(birdsong.content)
        # record.download(dest_dir=filepath)
        try:
            xc_recording.download(dest_dir=filepath)
        except Exception as e:
            print(f"Could not download url: {record_name}")
            print(e)
            continue
        num_downloaded += 1
        if num_downloaded == to_download: break
    return num_downloaded



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download using XenoCantoCollection metadata')

    # Add the arguments
    parser.add_argument('meta_data_path',
                           metavar='META_DIR',
                           type=str,
                           help='the path to original directory with .wav files')
    parser.add_argument('output_path',
                           metavar='OUT_DIR',
                           type=str,
                           help='the path to output directory to write new .wav/.png files')
    # parser.add_argument('-s', '--species',
    #                        metavar='S',
    #                        type=str,
    #                        help='specific species to use sliding window on',
    #                        default=None)
    parser.add_argument('-n', '--num',
                           metavar='NUM',
                           type=int,
                           help='maximum number of samples to download per species',
                           default=None)
    # Execute the parse_args() method
    args = parser.parse_args()
    download_bird_samples(args.meta_data_path, args.output_path, args.num)
