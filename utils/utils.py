# utils.py
# Author: Anshul Singh
# Date: 21 June 2024
# Update: 01 August 2024

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
from parselmouth.praat import call
import parselmouth

def list_subdirectories(path):
    """
    List all subdirectories in the given path.

    Parameters:
        path (str): The directory path.

    Returns:
        list: A list of subdirectory names.
    """
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {path}")

def list_files(path, extension='.wav'):
    """
    List all files in the given path with the specified extension.

    Parameters:
        path (str): The directory path.
        extension (str): The file extension to filter by.

    Returns:
        list: A list of file names.
    """
    try:
        return [f for f in os.listdir(path) if f.endswith(extension)]
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {path}")

def load_audio(audio_path, sr=None):
    """
    Load an audio file.

    Parameters:
        audio_path (str): The path to the audio file.
        sr (int, optional): The sample rate to use.

    Returns:
        tuple: A tuple containing the audio time series and the sample rate.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        return y, sr
    except Exception as e:
        raise Exception(f"Error loading audio file {audio_path}: {e}")

def make_segments(signal, segment_length, sr, min_percentage=0.8):
    """
    Segment the audio signal into smaller chunks.

    Parameters:
        signal (np.ndarray): The audio signal.
        segment_length (int): The length of each segment in seconds.
        sr (int): The sample rate of the audio signal.
        min_percentage (float): The minimum percentage of segment length for a segment to be included.

    Returns:
        pd.DataFrame: A DataFrame where each row is a segment of the audio signal.
    """
    min_length = int(segment_length * sr * min_percentage)
    segments = [signal[i:i + segment_length * sr] for i in range(0, len(signal), segment_length * sr)]
    return pd.DataFrame([seg for seg in segments if len(seg) >= min_length])

def reduce_noise(y, sr):
    """
    Reduce noise in the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        np.ndarray: The denoised audio signal.
    """
    y = librosa.effects.preemphasis(y)
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)
    noise_mag = np.mean(magnitude[:, :10], axis=1, keepdims=True)
    magnitude = np.maximum(magnitude - noise_mag, 0.0)
    y_denoised = librosa.istft(magnitude * phase)
    return y_denoised

def normalize_audio(y):
    """
    Normalize the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.

    Returns:
        np.ndarray: The normalized audio signal.
    """
    return y / np.max(np.abs(y))

def standardize_audio(y):
    """
    Standardize the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.

    Returns:
        np.ndarray: The standardized audio signal.
    """
    return (y - np.mean(y)) / np.std(y)

def preprocess_audio(y, sr):
    """
    Preprocess the audio signal by denoising, normalizing, and standardizing.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        np.ndarray: The preprocessed audio signal.
    """
    try:
        y_denoised = reduce_noise(y, sr)
        y_normalized = normalize_audio(y_denoised)
        y_standardized = standardize_audio(y_normalized)
        return y_standardized
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {e}")

def extract_features(y, sr):
    """
    Extract various features from the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        dict: A dictionary of extracted features.
    """
    features = {
        'F0Mean': get_f0_mean(y, sr),
        'F0Max': get_f0_max(y, sr),
        'F0Min': get_f0_min(y, sr),
        'F0Range': get_f0_range(y, sr),
        'Q25': np.percentile(y, 25),
        'Q50': np.percentile(y, 50),
        'Q75': np.percentile(y, 75),
        'Fpeak': np.argmax(np.abs(y)),
        'Sound Duration': librosa.get_duration(y=y, sr=sr),
        'AMVar': np.std(y) / np.mean(y),
        'AMRate': len(librosa.zero_crossings(y, pad=False)) / librosa.get_duration(y=y, sr=sr),
    }
    return features

def get_f0_mean(y, sr):
    """
    Calculate the mean fundamental frequency (F0) of the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        float: The mean fundamental frequency.
    """
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return np.mean(f0)

def get_f0_max(y, sr):
    """
    Calculate the maximum fundamental frequency (F0) of the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        float: The maximum fundamental frequency.
    """
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return np.max(f0)

def get_f0_min(y, sr):
    """
    Calculate the minimum fundamental frequency (F0) of the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        float: The minimum fundamental frequency.
    """
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return np.min(f0)

def get_f0_range(y, sr):
    """
    Calculate the range of the fundamental frequency (F0) of the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.
        sr (int): The sample rate of the audio signal.

    Returns:
        float: The range of the fundamental frequency.
    """
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return np.max(f0) - np.min(f0)

def get_formant_mean(selected_file, formant_number):
    """
    Calculate the mean formant frequency for a given formant number in the audio signal.

    Parameters:
        selected_file (str): The path to the audio file.
        formant_number (int): The formant number to calculate.

    Returns:
        float: The mean formant frequency.
    """
    try:
        sound = parselmouth.Sound(selected_file)
        formant = call(sound, "To Formant (burg)", 0.0, 5.0, 5500, 0.025, 50.0)
        formant_values = []
        for t in np.arange(0, sound.get_total_duration(), 0.01):
            formant_values.append(call(formant, "Get value at time", formant_number, t, 'Hertz', 'Linear'))
        return np.nanmean(formant_values)
    except Exception as e:
        raise Exception(f"Error calculating formant mean for formant {formant_number}: {e}")

def get_formant_dispersal(selected_file):
    """
    Calculate the dispersal (range) of formant frequencies in the audio signal.

    Parameters:
        selected_file (str): The path to the audio file.

    Returns:
        float: The dispersal of formant frequencies.
    """
    try:
        formant_means = []
        for i in range(1, 6):
            formant_means.append(get_formant_mean(selected_file, i))
        formant_means = [f for f in formant_means if not np.isnan(f)]
        return np.ptp(formant_means) if formant_means else np.nan
    except Exception as e:
        raise Exception(f"Error calculating formant dispersal: {e}")

def get_wiener_entropy_mean(y):
    """
    Calculate the mean Wiener entropy (spectral flatness) of the audio signal.

    Parameters:
        y (np.ndarray): The audio signal.

    Returns:
        float: The mean Wiener entropy.
    """
    try:
        spectrum = np.abs(librosa.stft(y))
        spectral_flatness = librosa.feature.spectral_flatness(S=spectrum)
        return np.mean(spectral_flatness)
    except Exception as e:
        raise Exception(f"Error calculating Wiener entropy mean: {e}")
