# Keras-music-generation
This program is based on the tutorial: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

Dependencies:

You need to install this list of library:
- Keras
- Tensorflow
- music21
- pickle
- csv
- glob
- tqdm
- midi2audio
- midi

How to use:

Put some .midi file in normal_songs directory then execute train.py to train the AI with this command line:
> python3 train.py

Keep last updates of each instruments were generated in updates directory and
rename them "instrument_name.hdf5". 
Example:
> Electric Bass-epoch50-loss1.04-updates.hdf5 -> Electric Bass.hdf5

Then execute predict.py with this command line to launch prediction:
> python3 predict.py

Some results will be obtained in music directory.

Finally execute merge.py by the following command line:
> python3 merge.py

It generates a .midi file called TotalSound.mid.

Enjoy the sound!!!
