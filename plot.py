#!/usr/bin/python

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


# Function to convert Hz to Mel
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700.0)

# Function to convert Mel to Hz
def mel_to_hz(m):
    return 700 * (10**(m / 2595.0) - 1)



def plot_features(direct, reverb, DRR, stpACC, fs, hop_len):
    
    num_time_bins, num_mel_bins = direct.shape
    duration = num_time_bins * hop_len / fs  # seconds
    f_min, f_max = 0, fs / 2  # Example: 0 Hz to 12 kHz

    # Generate mel bin centers in Hz
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_bins = np.linspace(mel_min, mel_max, num_mel_bins)
    
    # this is to only show a few ticks
    desired_freqs_hz = np.array([500, 4000, 12000]) 
    desired_freqs_mel = hz_to_mel(desired_freqs_hz)  # Convert to Mel
    desired_bins = np.interp(desired_freqs_mel, mel_bins, np.arange(num_mel_bins))  # Map to mel bin indices
    
    # use this if you had a linear frequency scale instead, i.e. not mel
    #mel_bins = np.linspace(f_min, f_max, num_mel_bins)  

    
    plt.subplot(4,1,1)
    plt.imshow(np.swapaxes(direct, 0, 1), cmap=cm.magma, origin='lower', aspect='auto', interpolation='none', extent=[0, duration, 0, num_mel_bins])
    # Adjust y-axis ticks to mel-based frequencies
    plt.yticks(desired_bins, labels=[f"{f/1000}" for f in desired_freqs_hz])
    plt.ylabel('Freq (kHz)', fontsize=14)
    plt.tick_params(bottom=True, labelbottom=False)
    plt.title('Direct Log mel Spec')
    #plt.colorbar()

    plt.subplot(4,1,2)
    plt.imshow(np.swapaxes(reverb, 0, 1), cmap=cm.magma, origin='lower', aspect='auto', interpolation='none', extent=[0, duration, 0, num_mel_bins])
    # Adjust y-axis ticks to mel-based frequencies
    plt.yticks(desired_bins, labels=[f"{f/1000}" for f in desired_freqs_hz])
    plt.ylabel('Freq (kHz)', fontsize=14)
    plt.tick_params(bottom=True, labelbottom=False)
    plt.title('Reverberant Log mel Spec')
    #plt.colorbar()
    
    plt.subplot(4,1,3)
    plt.imshow(np.swapaxes(DRR, 0, 1), cmap=cm.magma, origin='lower', aspect='auto', interpolation='none', extent=[0, duration, 0, num_mel_bins])
    # Adjust y-axis ticks to mel-based frequencies
    plt.yticks(desired_bins, labels=[f"{f/1000}" for f in desired_freqs_hz])
    plt.ylabel('Freq (kHz)', fontsize=14)
    plt.tick_params(bottom=True, labelbottom=False)
    plt.title('DRR Features')
    #plt.colorbar()

    plt.subplot(4,1,4)
    plt.imshow(np.swapaxes(stpACC, 0, 1), cmap=cm.magma, origin='lower', aspect='auto', interpolation='none', extent=[0, duration, 0,21])
    plt.ylabel('LagTimes (ms)', fontsize=14)
    plt.title('stpACC features')
    #plt.colorbar()

    plt.xlabel('time (s)', fontsize=14)
    plt.tick_params(axis='x', labelsize=12)

    #plt.grid()
    plt.show()