import pandas as pd
import numpy as np
from scipy.signal import firwin, lfilter, hilbert
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt 

# Set Variables for reading
df = pd.read_csv('./testrf3-200msa.csv', header=None, names=['time','wireless','spi_mod'])
fsampling = 200e6

# demean_normalize sets mean to 0 and standard deviation to 1
def demean_normalize(array):
    demeaned = array - np.mean(array)
    normalized = demeaned / np.std(demeaned)
    return normalized

# Importing Data
time = df['time'].to_numpy()
wireless = df['wireless'].to_numpy() 

# Spectral Analysis to identify the carrier frequency
wireless_fft = fft(wireless)
n = len(wireless)  # Length of the signal
freq = fftfreq(n, 1/fsampling)  # Frequency axis
magnitude_spectrum = np.abs(wireless_fft)
max_index = np.argmax(magnitude_spectrum)
carrier_frequency = freq[max_index]

# Setting Cutoffs for Narrow Band Filtering for Hilbert Transform
lowcut = (3/4)*carrier_frequency  # Low cut frequency in Hz (10 MHz)
highcut = (4/3)*carrier_frequency  # High cut frequency in Hz (20 MHz)
numtaps = 500  # Number of taps in the filter

# Designing the FIR bandpass filter
bandpass_fir = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fsampling)
a = 1

# Printing statistics for AM signal
print("The length of the signal is:" + str(n))
print("The carrier frequency is:" + str(carrier_frequency))

# Filtering Signal for Hilbert Transform
filtered_signal = lfilter(bandpass_fir,a,wireless)
filtered_signal_norm = demean_normalize(filtered_signal)

# Hilbert Transform
analytic_signal = hilbert(filtered_signal_norm)
envelope = np.abs(analytic_signal)

# Exporting 
np.savetxt("wireless_envelope.txt", demean_normalize(envelope), fmt='%f')
