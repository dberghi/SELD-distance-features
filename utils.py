#!/usr/bin/python

import numpy as np
import librosa
import scipy
from nara_wpe.wpe import wpe



def extractLogMelSpec(stft, fs, n_fft, n_mels, fmin, fmax):
    melW = librosa.filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax).T
    
    mel_spectrogram = np.dot(stft, melW)
    logmel_spectrogram = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-100, top_db=None)

    return logmel_spectrogram[:-1,:] # discart last sample


def extractDirectAndReverb(w, n_fft, hop_length): 
    """
    Extract direct and reverberant components from omni audio channel (w) with WPE algorithm
    Returns the STFTs of the direct and reverberant components
    """
    W = librosa.core.stft(
        y=w,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann',
        center=True,
        dtype=np.complex64,
        pad_mode='reflect'
    ).T
    W = np.expand_dims(W, 1)
    W = W.transpose(2,1,0) # needed for this particular wpe implementation

    # you can play and find your suitable hyperparameters
    D = wpe(
        W,
        taps=60,
        delay=5,
        iterations=5,
        statistics_mode='full'
    ) 
    D = np.squeeze(D, axis=1)
    
    direct = librosa.istft(
        D,
        hop_length=hop_length,
        win_length=n_fft, 
        n_fft=n_fft, 
        window='hann', 
        center=True, 
        dtype=None, 
    )
    # temporal difference
    reverb = w - direct

    R = librosa.core.stft(
        y=reverb,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann', #np.hanning(n_fft),
        center=True,
        dtype=np.complex64,
        pad_mode='reflect'
    ).T

    D = D.transpose(1,0)
    return D, R 
    
def extract2DstpACC(x, n_fft, hop_length, downsample=True, ds_factor=8): 
    """
    Returns the 2D stpACC features of the input signal x via frequency analysis
    """
    Px = librosa.stft(y=x,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      center=True,
                      window=np.hanning(n_fft),
                      pad_mode='reflect')
    

    R = Px * np.conj(Px)
    
    n_frames = R.shape[1]
    stp_acc = []
    #acc2D = []

    
    for i in range(n_frames):
        spec = R[:,i].flatten()
        acc = np.fft.irfft(spec)
        acc = acc[:len(acc) // 2]
        acc = np.clip(acc, 1e-100, acc.max()) # this is needed to avoid dividing by 0
        # norm
        acc = acc / max(abs(acc))
        stp = np.convolve(acc**2, np.hanning(8), 'same') # 8 samples at 24kHz ~= 3.3e-4sec => 10cm
        # norm
        stp = stp / max(stp)
        # Downsample
        if downsample:
            stp = scipy.signal.resample(stp, num=round(len(stp) / ds_factor))
        stp_acc.append(stp)
       
    stp_acc = np.array(stp_acc)
    
    return stp_acc[:-1,:] 


