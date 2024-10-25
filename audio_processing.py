import scipy.io.wavfile as wav
import numpy as np
import pyloudnorm
import scipy
from scipy.spatial.distance import cdist
import pandas as pd


def load_wav(file_path: str) -> (float, np.array):
    """
    Loads a .wav file from the given file_path as a numpy array. Combines stereo into mono.
    Returns:
        sample_rate: The sample rate of the file
        sound: The file represented as a numpy array
    """
    sample_rate, sound = wav.read(file_path)

    if len(sound.shape) > 1:
        sound = sound.mean(axis=1)

    return sample_rate, sound


def save_wav(destination_path: str, sample_rate: int or float, sound: np.array) -> None:
    if sound.max() > 1 or sound.min() < -1:
        sound = db_to_amplitude(sound)

    wav.write(destination_path, int(sample_rate), sound)


def amplitude_to_db(sound: np.array) -> np.array:
    """
    Converts a .wav array from the linear amplitude normally used in .wav files to decibels.
    """
    # Avoid log of zero by adding a small epsilon value
    epsilon = 1e-10
    return 20 * np.log10(np.abs(sound) + epsilon)


def db_to_amplitude(sound: np.array) -> np.array:
    return 10 ** (sound / 20)


def normalize_volumes(sound1: np.array, sound2: np.array, sample_rate: float) -> np.array:
    """
    Normalizes the volume of the sound so that the perceived loudness of the louder file is reduced to match the
    perceived loudness of the second file.
    Params:
        sound1: The first .wav array to be potentially normalized
        sound2: The other .wav array to be potentially normalized
        sample_rate: The sample rate of the files (must match)
    """
    # Convert audio data to floating point values between -1 and 1
    sound1 = sound1.astype(np.float32) / np.max(np.abs(sound1))
    sound2 = sound2.astype(np.float32) / np.max(np.abs(sound2))

    # TODO - this should probably not be this small for larger files
    meter = pyloudnorm.Meter(sample_rate, block_size=0.2)

    sound1_lufs = meter.integrated_loudness(sound1)
    sound2_lufs = meter.integrated_loudness(sound2)

    if sound1_lufs > sound2_lufs:
        sound1 = pyloudnorm.normalize.loudness(sound1, sound1_lufs, sound2_lufs)

    elif sound2_lufs > sound1_lufs:
        sound2 = pyloudnorm.normalize.loudness(sound2, sound2_lufs, sound1_lufs)

    return sound1, sound2


def align_sound_data_at_start(sound1: np.array, sound2: np.array, threshold: float = 0.05) -> (np.array, np.array):
    """
    Takes two .wav numpy arrays and aligns them at the start of the first note. Assumes that there is sufficient
    padding before the note begins. Also assumes that the sound arrays are represented in decibels.

    Params:
        sound1: A numpy array representation of a .wav file, represented in decibels.
        sound2: A numpy array representation of a .wav file, represented in decibels.
        threshold: The threshold that must be met for the note to be considered begun.
    Returns:
        (sound_data1_aligned, sound_data2_aligned): A 2-tuple containing aligned versions of the arrays.
        TODO - look into adding a parameter that controls whether copies are made or the arrays are directly modified.
        TODO - look into adding quantization, so that the other notes can be aligned as well.
        TODO - look into dBFS
    """
    # Find first occurrence of exceeded threshold in each sound array.
    start1 = np.argmax(sound1 > threshold)
    start2 = np.argmax(sound2 > threshold)

    if start1 < start2:
        padding = start2 - start1
        sound1 = np.pad(sound1, (padding, 0), "constant")

    elif start2 < start1:
        padding = start1 - start2
        sound2 = np.pad(sound2, (padding, 0), "constant")

    max_sound_length = min(len(sound1), len(sound2))

    if len(sound1) > len(sound2):
        sound1 = sound1[:max_sound_length]

    elif len(sound2) > len(sound1):
        sound2 = sound2[:max_sound_length]

    return sound1, sound2


def match_peaks(peaks1: np.array, peaks2: np.array, tolerance=50) -> np.array:
    # Compute all distances between peaks in time (sample indices)
    distances = cdist(peaks1.reshape(-1, 1), peaks2.reshape(-1, 1), metric="euclidean")

    # Find pairs of peaks where the distance is within the tolerance
    matched_peaks1 = []
    matched_peaks2 = []

    for i in range(len(peaks1)):
        # Find the closest peak in peaks2 that is within tolerance
        closest_peak_idx = np.argmin(distances[i])  # Find the index of the closest peak in peaks2
        if distances[i, closest_peak_idx] <= tolerance:  # If it's within the tolerance
            matched_peaks1.append(peaks1[i])  # Add the peak from peaks1
            matched_peaks2.append(peaks2[closest_peak_idx])  # Add the corresponding peak from peaks2

    return np.array(matched_peaks1), np.array(matched_peaks2)


def calculate_total_peak_distance(peaks1: np.array, peaks2: np.array, shift: int) -> np.array:
    # Apply the shift to peaks1
    shifted_peaks1 = peaks1 + shift

    # Match peaks based on proximity
    matched_peaks1, matched_peaks2 = match_peaks(shifted_peaks1, peaks2)

    # Calculate total distance between matched peaks
    total_distance = np.sum(np.abs(matched_peaks1 - matched_peaks2))

    return total_distance


def align_sound_data_by_peaks(sound1: np.array, sound2: np.array, threshold: float = 0.9) -> (np.array, np.array):
    """
    Just using the start of the signal usually won't fully align the signals, because the length of the string attack
    can vary based on many factors:
      - The length of the pick/finger scraping against the string
      - The length of time the compressor is configured to wait before applying compression (if applicable)
      - String gauge
      - How well the bridge sustains
      - etc.
    So we try to find the shift amount that best aligns the two signals by aligning them at their peak amplitudes.
    """
    peaks1, properties1 = scipy.signal.find_peaks(
        sound1,
        height=0.1
    )

    peaks1_heights = properties1["peak_heights"]

    peaks2, properties2 = scipy.signal.find_peaks(
        sound2,
        height=0.1,
    )

    peaks2_heights = properties2["peak_heights"]

    sound1_max_peak = peaks1[np.argmax(peaks1_heights)]
    sound2_max_peak = peaks2[np.argmax(peaks2_heights)]

    # Shift the earlier signal right by padding the front with zeros.
    if sound1_max_peak < sound2_max_peak:
        padding = sound2_max_peak - sound1_max_peak
        print(f"padding sound1 with {padding / 44100} sec")
        sound1 = np.pad(sound1, (padding, 0), "constant")

    elif sound2_max_peak < sound1_max_peak:
        padding = sound1_max_peak - sound2_max_peak
        print(f"padding sound2 with {padding / 44100} sec")
        sound2 = np.pad(sound2, (padding, 0), "constant")

    else:
        print("wtf")

    # Once the signals have been aligned, trim the excess from whichever signal is longer.
    max_sound_length = min(len(sound1), len(sound2))

    if len(sound1) > len(sound2):
        sound1 = sound1[:max_sound_length]

    elif len(sound2) > len(sound1):
        sound2 = sound2[:max_sound_length]

    return sound1, sound2


def prepare_sounds_for_comparison(sound1: np.array, sound2: np.array, sample_rate: float) -> (np.array, np.array):
    """
    Performs a sequence of preprocessing steps to prepare the sound arrays for comparison.
        1. Normalizes the volume of the sounds by using perceived loudness, reducing the volume of the louder file
           so that it matches the perceived loudness of the quieter file.
        2. Pads and trims the sounds so that they begin at the same time and have the same length.
    Params:
        sound1: The first .wav array.
        sound2: The other .wav array.
    Returns:
        (sound1_preprocessed, sound2_preprocessed): A 2-tuple containing preprocessed versions of the .wav arrays that
        are ready to be compared.
    """
    sound1, sound2 = normalize_volumes(sound1, sound2, sample_rate)
    sound1, sound2 = align_sound_data_by_peaks(sound1, sound2)

    return sound1, sound2
