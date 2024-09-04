import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import fft, fftfreq
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Assuming features_df is already created and contains the 'label' column
def plot_feature_distributions(features_df, features):
    """Plots individual feature distributions by class."""
    for feature in features:
        if feature == 'label':
            continue  # Skip plotting the label column
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='label', y=feature, data=features_df)
        plt.title(f'Distribution of {feature} by Class')
        plt.ylabel('Value')
        plt.xlabel('Class')
        plt.savefig(f'feature_distribution_{feature}.png')  # Save each plot as a PNG file
        plt.show()

def plot_sampled_windows(wireless_env, edges, labels, num_windows):
    """Plot a sample of the windows with their corresponding labels."""
    plt.figure(figsize=(15, 5 * num_windows))
    
    sample_indices = np.random.choice(range(len(edges)), num_windows, replace=False)
    for i, idx in enumerate(sample_indices):
        edge = edges[idx]
        start = int(edge - half_samples)
        end = int(edge + half_samples)
        
        if start < 0 or end > len(wireless_env):  # Ensure index is within bounds
            continue
        
        plt.subplot(num_windows, 1, i + 1)
        plt.plot(wireless_env[start:end], label=f'Label: {labels[idx]}')
        plt.title(f'Window {i+1} around edge {edge} with label {labels[idx]}')
        plt.xlabel('Sample index')
        plt.ylabel('Signal amplitude')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

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

def apply_and_save_pca(features_df, n_components=0.95, output_file='reduced_features.csv'):
    # Assuming all columns except 'label' are features
    features = features_df.loc[:, features_df.columns != 'label']
    labels = features_df['label']
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Initialize PCA
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)
    
    # Convert the reduced features back to a DataFrame
    features_reduced_df = pd.DataFrame(features_reduced, columns=[f'PC{i+1}' for i in range(features_reduced.shape[1])])
    features_reduced_df['label'] = labels
    
    # Save the reduced features to a CSV file
    features_reduced_df.to_csv(output_file, index=False)
    
    return features_reduced_df, pca

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
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.legend()

# Plotting the Wireless signal
plt.subplot(3, 1, 3)  # 3 rows, 1 column, third plot
plt.plot(df_ten_percent['time'], df_ten_percent['wireless'], label='Wireless Signal')
plt.title('Wireless Signal')
plt.xlabel('Time (ms)')
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

fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
axs[0].plot(df['time'][:ten_percent_length], wireless_env[:ten_percent_length], label='Wireless Envelope')
axs[0].set_title('Wireless Envelope and Falling Edges')
axs[0].set_ylabel('Amplitude')

# Second subplot for digitized SPI data
axs[1].plot(df['time'][:ten_percent_length], spi_data[:ten_percent_length], label='Digitized SPI Data', color='orange')
axs[1].set_title('Digitized SPI Data and Falling Edges')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Digital Signal')

# Mark falling edges on the first subplot
for edge in falling_edges:
    if edge < ten_percent_length:  # Ensure edge is within the first 10%
        axs[1].axvline(x=df['time'].iloc[edge], color='r', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.95)
plt.show()

# Extract features and labels for each window
feature_label_sets = []
for index, edge in enumerate(falling_edges):
    start = int(edge - samples_per_period)
    end = int(edge + samples_per_period)
    if start < 0 or end > len(wireless_env):  # Check to ensure indices are within bounds
        continue  # Skip this window to avoid indexing errors
    window = wireless_env[start:end]
    if len(window) > 0:
        features = extract_features(window)
        features['label'] = true_labels[index]
        feature_label_sets.append(features)

# Convert feature-label sets to DataFrame for easier analysis and visualization
features_df = pd.DataFrame(feature_label_sets)

plot_sampled_windows(wireless_env, falling_edges, true_labels, num_windows=5)

# Save features to a CSV file
features_df.to_csv("extracted_features_with_labels.csv", index=False)

# Call the function to plot features
features_to_plot = [col for col in features_df.columns if col != 'label']
# plot_feature_distributions(features_df, features_to_plot)

features_reduced_df, pca = apply_and_save_pca(features_df, output_file='reduced_features.csv')
