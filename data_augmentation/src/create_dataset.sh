#!/bin/bash
mkdir data
cd data
mkdir call_download
mkdir song_download
mkdir call_filtered
mkdir song_filtered
mkdir call_timeshift
mkdir song_timeshift
mkdir call_full_aug
mkdir song_full_aug
mkdir call_spectrograms
mkdir song_spectrograms
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
cd ..
python lglik/Code/birds/src/recording_download.py -f data/call_download/ data/call_filtered/
python lglik/Code/birds/src/recording_download.py -f data/song_download/ data/song_filtered/
python lglik/Code/birds/data_augmentation/src/WavSpoofer.py -ts data/call_filtered/ data/call_timeshift/
python lglik/Code/birds/data_augmentation/src/WavSpoofer.py -ts data/song_filtered/ data/song_timeshift/
python lglik/Code/birds/data_augmentation/src/WavSpoofer.py -cv data/call_timeshift/ data/call_full_aug/
python lglik/Code/birds/data_augmentation/src/WavSpoofer.py -cv data/song_timeshift/ data/song_full_aug/
cp -a data/call_filtered/. data/call_full_aug
cp -a data/song_filtered/. data/song_full_aug
python lglik/Code/birds/src/recording_download.py -s data/call_full_aug/ data/call_spectrograms/
python lglik/Code/birds/src/recording_download.py -s data/song_full_aug/ data/song_spectrograms/
cp -a data/call_filtered/. data/all_spectrograms
cd all_spectrograms
mv Amaziliadecora* AmaziliadecoraCall
mv ..............*  .............Call
rm *Cat*
rm *Emp*
cd ../../
cp -a data/song_filtered/. data/all_spectrograms
cd all_spectrograms
mv Amaziliadecora* AmaziliadecoraSong
mv ..............*  .............Song
rm *Cat*
rm *Emp*
cd ../../
python lglik/Code/birds/data_augmentation/src/CreateTrainValidation.py data/all_spectrograms/