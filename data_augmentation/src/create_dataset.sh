#/usr/bin/env sh

# the filepath to the call samples to use
call_path=/home/data/birds/NEW_BIRDSONG/CALL/

# the filepath to the song samples to use
song_path=/home/data/birds/NEW_BIRDSONG/SONG/

# the filepath to the local repo to use
code_path=/home/lglik/Code/birds

# the filepath for where the data set should be created
out_path=.

mkdir $out_path/data
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
mkdir train
cp -a all_spectrograms/. train
mkdir validation
cp -a all_spectrograms/. validation
cp -a $call_path. call_download
cp -a $song_path. song_download
python $code_path/src/recording_download.py -f call_download call_filtered
python $code_path/src/recording_download.py -f song_download song_filtered
python $code_path/data_augmentation/src/WavSpoofer.py -ts call_filtered call_timeshift
python $code_path/data_augmentation/src/WavSpoofer.py -ts song_filtered song_timeshift
python $code_path/data_augmentation/src/WavSpoofer.py -cv call_timeshift call_full_aug
python $code_path/data_augmentation/src/WavSpoofer.py -cv song_timeshift song_full_aug
cp -a call_filtered/. call_full_aug
cp -a song_filtered/. song_full_aug
python $code_path/src/recording_download.py -s call_full_aug call_spectrogram
python $code_path/src/recording_download.py -s song_full_aug song_spectrogram
cp -a call_spectrogram/. all_spectrograms
cd all_spectrograms
mv Ama*.jpg Ama*Call
mv Arr*.jpg Arr*Call
mv Cor*.jpg Cor*Call
mv Dys*.jpg Dys*Call
mv Eup*.jpg Eup*Call
mv Hen*.jpg Hen*Call
mv Hyl*.jpg Hyl*Call
mv Lop*.jpg Lop*Call
mv Tangarag*.jpg Tangarag*Call
mv Tangarai*.jpg Tangarai*Call
rm *Cat*.jpg
rm *Emp*.jpg
cd ..
cp -a song_spectrogram/. all_spectrograms
cd all_spectrograms
mv Ama*.jpg Ama*Song
mv Arr*.jpg Arr*Song
mv Cor*.jpg Cor*Song
mv Dys*.jpg Dys*Song
mv Eup*.jpg Eup*Song
mv Hen*.jpg Hen*Song
mv Hyl*.jpg Hyl*Song
mv Lop*.jpg Lop*Song
mv Tangarag*.jpg Tangarag*Song
mv Tangarai*.jpg Tangarai*Song
rm *Cat*.jpg
rm *Emp*.jpg
cd ..
python $code_path/data_augmentation/src/CreateTrainValidation.py all_spectrograms
