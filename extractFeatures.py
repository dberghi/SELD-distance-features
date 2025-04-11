#!/usr/bin/python

import numpy as np
import soundfile as sf
import plot as plotFeat
from utils import extractDirectAndReverb, extract2DstpACC, extractLogMelSpec


audio_path = 'clip.wav'
amin = 1e-100

# Set metaparameters for log mel spectrograms & DRR features
n_fft = 512
hop_length = 150
n_mels = 64

# Set metaparameters for short-term power of the autocorrelation
stp_n_fft = 1024
stp_hop_length = 150
downsample = True
ds_factor = 8 # downsampling factor



def main():
    audio, fs = sf.read(audio_path)
    # if audio is FOA format extract W channel (comment the next two lines if audio is mono already)
    audio = np.swapaxes(audio, 0, 1)
    audio = audio[0]
    
    # Extract short-term power of the autocorrelation features
    stpACC = extract2DstpACC(audio, n_fft=stp_n_fft, hop_length=stp_hop_length, downsample=downsample, ds_factor=ds_factor) # (T, n_mel)
    
    # Extract STFT of direct and reverberant components
    dir, rev = extractDirectAndReverb(w=audio, n_fft=n_fft, hop_length=hop_length) # STFTs (T, n_fft/2+1)

    # you can either use dir and rev independently (DR features), or compute their ratio (DRR features)
    dir_logmel = extractLogMelSpec(dir, fs, n_fft, n_mels, fmin=0, fmax=fs//2) # (T, n_mel)
    rev_logmel = extractLogMelSpec(rev, fs, n_fft, n_mels, fmin=0, fmax=fs//2) # (T, n_mel)
    DR = np.stack((dir_logmel, rev_logmel), axis=0) # DR features # (2, T, n_mel)

    # compute power spectral densities
    dir_psd = abs(dir) ** 2
    rev_psd = abs(rev) ** 2
    # clamp to avoid very low or zero-energy that would lead to instability or undefined values
    # with pytorch can use torch.clamp() instead
    dir_psd = np.clip(dir_psd, a_min=1e-100, a_max=dir_psd.max())
    rev_psd = np.clip(rev_psd, a_min=1e-100, a_max=rev_psd.max())
    # direct-to-reverberant ratio
    DRR = dir_psd / rev_psd
    # use log mel space
    DRR = extractLogMelSpec(DRR, fs, n_fft, n_mels, fmin=0, fmax=fs//2) #(T, n_mel)

    plotFeat.plot_features(dir_logmel,
                           rev_logmel,
                           DRR,
                           stpACC,
                           fs=fs, hop_len=hop_length)

if __name__ == "__main__":
    main()
