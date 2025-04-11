# Distance Input Features for 3D SELD

This is a Python implementation of the feature extraction methods described in the paper "Reverberation-based Features for Sound Event Localization and Detection with Distance Estimation" (preprint).
> D. Berghi, P. J. B. Jackson. Reverberation-based Features for Sound Event Localization and Detection with Distance Estimation. ArXiv (preprint), 2025. [[**arXiv**]](https://arxiv.org/)

### Dependencies

This implementation mainly leverages commonly used libraries for audio processing. Make sure you have installed:
```
nara_wpe
numpy
scipy
librosa
matplotlib (optional)
```
We used the `nara_wpe` library by Drude et al. (https://github.com/fgnt/nara_wpe) to extract the direct sound component. You are free to use different methods, but make sure to acknowledge their work if you use `nara_wpe`.
The library can be easily installed with:
```
pip install nara_wpe
```
### Usage

`extractFeatures.py` will automatically extract and plot the features from `clip.wav`. The provided clip is a FOA snipped extracted from the STARSS23 dataset. You can load your desired audio file by changing `audio_path`.

Set your preferred log mel spectrogram hyperparameters. These will be used for DR and DRR features.
```
n_fft = 512
hop_length = 150
n_mels = 64
```

Set metaparameters for short-term power of the autocorrelation (stpACC) features.

We recommend considering time lags up to about 20ms, but you are free to pick different values. 
So, considering only positive time lags in the autocorrelation, choose stp_n_fft so that (stp_n_fft / 2) / fs ≈ 20ms
E.g. for fs=24kHz we used stp_n_fft=1024 

Downsampling the autocorrelation is optional, but useful to allow concatenation with, e.g., log-mel spectrograms.
For example, if you have log mel spectrograms with 64 mel bins, and have stp_n_fft/2 = 512 positive time lags, you should set ds_factor=8
```
stp_n_fft = 1024
stp_hop_length = 150
downsample = True
ds_factor = 8 # downsampling factor
```
