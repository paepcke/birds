import os
from os import listdir
from os.path import isfile, join
import random
starting_filepath = '/home/data/birds/NEW_BIRDSONG/current_files/'
birdlist =  os.listdir(starting_filepath)
print(birdlist)
if '.DS_STORE' in birdlist:
        birdlist.remove('.DS_Store') #ghost file
print(birdlist)
for bird in birdlist:
        for file in os.listdir(starting_filepath + bird):
                print(file)
                if random.random() >= 0.8:
                        os.rename(starting_filepath + bird + '/' + file, starting_filepath +
                                  "../validation/" + bird + "/" + file)
                else:
                        os.rename(starting_filepath + bird + '/' + file, starting_filepath +
                                  "../train/" + bird + "/" + file)
