import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import fft, fftfreq
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import pywt

def extract_features(segment):
    """ Extract relevant features for OOK ASK signal classification, including first derivative features. """
    features = {}

    if len(segment) > 1:  # Ensure there are enough data points for derivative calculation
        derivative = np.diff(segment, n=1)
        derivative = np.append(derivative, derivative[-1])  # Append last value to match the original length
    else:
        derivative = np.zeros_like(segment)

    # Smooth the derivative to reduce noise effects
    if len(derivative) > 5:  # Ensure enough points for smoothing
        derivative = savgol_filter(derivative, 5, 3)  # Window size 5, polynomial order 3

    # Time-domain features from the original signal
    features['std'] = np.std(segment)
    features['max'] = np.max(segment)
    features['min'] = np.min(segment)
    features['mean'] = np.mean(segment)

    # Time-domain features from the first derivative
    features['deriv_std'] = np.std(derivative)
    features['deriv_max'] = np.max(derivative)
    features['deriv_min'] = np.min(derivative)
    features['deriv_mean'] = np.mean(derivative)

    # Skewness and Kurtosis
    features['skewness'] = skew(segment)
    features['kurtosis'] = kurtosis(segment)
    features['deriv_skewness'] = skew(derivative)
    features['deriv_kurtosis'] = kurtosis(derivative)

    # Frequency-domain features using FFT
    freq_domain = fft(segment)
    power_spec = np.abs(freq_domain) ** 2
    features['fft_peak_freq'] = np.argmax(power_spec[:len(power_spec)//2])
    features['fft_peak_power'] = np.max(power_spec)

    # Wavelet-based features for multi-level analysis
    coeffs = pywt.wavedec(segment, 'db4', level=5)
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_avg_coeff_{i}'] = np.mean(coeff)
        features[f'wavelet_std_coeff_{i}'] = np.std(coeff)
        features[f'wavelet_skew_coeff_{i}'] = skew(coeff)
        features[f'wavelet_kurt_coeff_{i}'] = kurtosis(coeff)
        features[f'wavelet_energy_coeff_{i}'] = np.sum(coeff**2)

    return features


# Load the data
df = pd.read_csv('./testrf4-200msa.csv', header=None, names=['time', 'spi_data', 'wireless', 'spi_clock'])
fsampling = 200e6  # Sampling rate

# Calculate 10% of the length of the data
ten_percent_length = int(len(df) * 0.1)

# Select the first 10% of the data
df_ten_percent = df.iloc[:ten_percent_length]

# Plot each signal in a separate plot
plt.figure(figsize=(12, 8))  # Set the size of the entire figure containing all plots

# Plotting the SPI clock signal
plt.subplot(3, 1, 1)  # 3 rows, 1 column, first plot
plt.plot(df_ten_percent['time'], df_ten_percent['spi_clock'], label='SPI Clock')
plt.title('SPI Clock Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plotting the SPI data signal
plt.subplot(3, 1, 2)  # 3 rows, 1 column, second plot
plt.plot(df_ten_percent['time'], df_ten_percent['spi_data'], label='SPI Data')
plt.title('SPI Data Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plotting the Wireless signal
plt.subplot(3, 1, 3)  # 3 rows, 1 column, third plot
plt.plot(df_ten_percent['time'], df_ten_percent['wireless'], label='Wireless Signal')
plt.title('Wireless Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()  # Adjust layout to make it look nice
plt.show()  # Show the plots

spi_data = df["spi_data"].to_numpy()
spi_clock = df["spi_clock"].to_numpy()
wireless = df["wireless"].to_numpy()
wireless_env = np.loadtxt("wireless_envelope.txt")

spi_clock_fft = fft(spi_clock)
n = len(spi_clock)  # Length of the signal
freq = fftfreq(n, 1/fsampling)  # Frequency axis
magnitude_spectrum = np.abs(spi_clock_fft)

# Filter frequencies above 100 kHz
high_freq_indices = np.where(freq > 100000)  # Indices of frequencies above 100 kHz
filtered_freq = freq[high_freq_indices]
filtered_magnitude = magnitude_spectrum[high_freq_indices]

# Find maximum amplitude in the filtered range
max_index = np.argmax(filtered_magnitude)
max_frequency = filtered_freq[max_index]
max_amplitude = filtered_magnitude[max_index]

period_seconds = 1 / max_frequency  # Period in seconds
samples_per_period = fsampling * period_seconds  # Number of samples per period

print("Max Amplitude Frequency of spi_clock:", max_frequency, "Hz")
print("Amplitude at Max Frequency:", max_amplitude)

# Digitizing spi_clock and spi_data
threshold = 0.65
spi_data = np.where(spi_data < threshold, 0, 1)
spi_clock = np.where(spi_clock < threshold, 0, 1)

# Find falling edges of spi_clock
falling_edges = np.where((spi_clock[:-1] == 1) & (spi_clock[1:] == 0))[0]

# Sample spi_data at falling edges
true_labels = spi_data[falling_edges]

# Calculate half of the samples_per_period
half_samples = samples_per_period // 2

# Extract features and labels for each window
feature_label_sets = []
for index, edge in enumerate(falling_edges):
    start = int(edge - half_samples)
    end = int(edge + half_samples)
    if start < 0 or end > len(wireless_env):  # Check to ensure indices are within bounds
        continue  # Skip this window to avoid indexing errors
    window = wireless_env[start:end]
    if len(window) > 0:
        features = extract_features(window)
        features['label'] = true_labels[index]
        feature_label_sets.append(features)

# Convert feature-label sets to DataFrame for easier analysis and visualization
features_df = pd.DataFrame(feature_label_sets)

# Save features to a CSV file
features_df.to_csv("extracted_features_with_labels.csv", index=False)

print("Features and labels extracted and saved. Total windows processed:", len(features_df))
