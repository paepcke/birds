#/usr/bin/env sh

# the filepath to the call samples to use
call_path=/home/data/birds/NEW_BIRDSONG/CALL/

# the filepath to the song samples to use
song_path=/home/data/birds/NEW_BIRDSONG/SONG/

# the filepath to the local repo to use
code_path=/home/lglik/Code/birds

# the filepath for where the data set should be created
out_path=band_and_norm

# make all necessary directories
mkdir $out_path/data
# create folders for processing the samples
cd $out_path/data
mkdir call_download
mkdir song_download
mkdir call_filtered
mkdir song_filtered
mkdir call_timeshift
mkdir song_timeshift
mkdir call_full_aug
mkdir song_full_aug
mkdir call_spectrogram
mkdir song_spectrogram
mkdir all_spectrograms
# create folders for storing the processed samples
cd all_spectrograms
mkdir AmaziliadecoraCall
mkdir AmaziliadecoraSong
mkdir ArremonaurantiirostrisCall
mkdir ArremonaurantiirostrisSong
mkdir CorapipoalteraCall
mkdir CorapipoalteraSong
mkdir DysithamnusmentalisCall
mkdir DysithamnusmentalisSong
mkdir EuphoniaimitansCall
mkdir EuphoniaimitansSong
mkdir HenicorhinaleucostictaCall
mkdir HenicorhinaleucostictaSong
mkdir HylophilusdecurtatusCall
mkdir HylophilusdecurtatusSong
mkdir LophotriccuspileatusCall
mkdir LophotriccuspileatusSong
mkdir TangaragyrolaCall
mkdir TangaragyrolaSong
mkdir TangaraicterocephalaCall
mkdir TangaraicterocephalaSong
cd ..
# make train and validation folders and populate them with empty directories from all_spectrograms
mkdir train
cp -a all_spectrograms/. train
mkdir validation
cp -a all_spectrograms/. validation
# copy source song and call samples into call_download and song_download
cp -a $call_path. call_download
cp -a $song_path. song_download
# filter the samples
python $code_path/src/recording_download.py -f call_download call_filtered
python $code_path/src/recording_download.py -f song_download song_filtered
# time shift and change the volume of a copy of the samples
python $code_path/data_augmentation/src/WavSpoofer.py -ts call_filtered call_timeshift
python $code_path/data_augmentation/src/WavSpoofer.py -ts song_filtered song_timeshift
python $code_path/data_augmentation/src/WavSpoofer.py -cv call_timeshift call_full_aug
python $code_path/data_augmentation/src/WavSpoofer.py -cv song_timeshift song_full_aug
# put the augmented and unaugmented samples in the same folder
cp -a call_filtered/. call_full_aug
cp -a song_filtered/. song_full_aug
# convert samples to spectrograms
python $code_path/src/recording_download.py -s call_full_aug call_spectrogram
python $code_path/src/recording_download.py -s song_full_aug song_spectrogram
# copy and then move call spectrograms into folders based on their species
cp -a call_spectrogram/. all_spectrograms
cd all_spectrograms
mv *Ama*.png Ama*Call
mv *Arr*.png Arr*Call
mv *Cor*.png Cor*Call
mv *Dys*.png Dys*Call
mv *Eup*.png Eup*Call
mv *Hen*.png Hen*Call
mv *Hyl*.png Hyl*Call
mv *Lop*.png Lop*Call
mv *Tangarag*.png Tangarag*Call
mv *Tangarai*.png Tangarai*Call
rm *Cat*.png
rm *Emp*.png
cd ..
# copy and then move call spectrograms into folders based on their species
cp -a song_spectrogram/. all_spectrograms
cd all_spectrograms
mv *Ama*.png Ama*Song
mv *Arr*.png Arr*Song
mv *Cor*.png Cor*Song
mv *Dys*.png Dys*Song
mv *Eup*.png Eup*Song
mv *Hen*.png Hen*Song
mv *Hyl*.png Hyl*Song
mv *Lop*.png Lop*Song
mv *Tangarag*.png Tangarag*Song
mv *Tangarai*.png Tangarai*Song
rm *Cat*.png
rm *Emp*.png
cd ..
# split the data into a training set and a validation set
python $code_path/data_augmentation/src/CreateTrainValidation.py all_spectrograms
