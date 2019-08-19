# claps-classifier
A model for detecting and classifying hand claps in audio files  
There are pre-trained Keras model for classifying short segment of audio using Mel-frequency cepstral coefficients (MFCC).
Dataset [FSDKaggle2018](https://zenodo.org/record/2552860#.XVsDSLr7RhG) was used for training, and it's possible to reuse 
this model for searching another patterns just by training it on other data and replacing pattern files.

- For now it fails on noisy files, maybe should add some kind of filtering or try to train on noisy data
