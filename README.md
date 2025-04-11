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
