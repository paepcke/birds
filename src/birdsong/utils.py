from datetime import datetime
# Given input format mm:ss, converts to total seconds
def time_to_seconds(time_str):
    pt = datetime.strptime(time_str,'%M:%S')
    total_seconds = pt.second + pt.minute*60
    return total_seconds
# Function takes in a XenoCantoCollection object and returns a DataFrame
# with indexes the species and total_recording_length for that species
def recording_lengths_by_species(collection):
    total_rec_len = 0
    for rec in collection(one_per_bird_phylo=False):
        total_rec_len += time_to_seconds(rec.length)




if __name__ == '__main__':
    main()
