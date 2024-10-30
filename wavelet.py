import numpy as np
import pycwt
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
import matplotlib.pyplot as plt
import matplotlib.scale
from matplotlib.colors import SymLogNorm
import time
from numba import njit, prange


def calculate_power_spectrum(sound: np.array, sample_rate: int or float, start_frequency: int, end_frequency: int) -> np.array:
    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
        sound,
        dt=1 / sample_rate,
        wavelet=pycwt.Morlet(3),
        freqs=np.array([i for i in range(start_frequency, end_frequency)])
    )
    
    wave_power = np.abs(wave) ** 2
    
    return wave_power


@njit
def get_abs_diff(a1: np.array, a2: np.array) -> (float, float):
    diff_sum = 0.0
    diff_square_sum = 0.0
    num_items = a1.shape[0] * a1.shape[1]

    for i in range(a1.shape[0]):
        for j in range(a1.shape[1]):
            abs_diff = abs(a1[i, j] - a2[i, j])

            diff_sum += abs_diff
            diff_square_sum += (abs_diff ** 2)

    mean = diff_sum / num_items

    var = (diff_square_sum / num_items) - (mean ** 2)
    std = np.sqrt(var)

    return mean, std


@njit(parallel=True)
def _find_optimal_alignment(a1: np.array, a2: np.array, num_shifts: int) -> (int, float, float):
    means = np.array([float("inf") for _ in range(num_shifts)])
    stds = np.array([float("inf") for _ in range(num_shifts)])

    for i in prange(num_shifts):
        # Truncate each row, so that spectrum 1's length does not exceed spectrum 2's length.
        start = num_shifts - i

        # num_shifts is the length of spectrum2
        end = start + num_shifts

        means[i], stds[i] = get_abs_diff(a1[start: end, :], a2)

    spectrum1_best_shift = np.argmin(means)

    return spectrum1_best_shift, means[spectrum1_best_shift], stds[spectrum1_best_shift]


def pad_spectrum(a: np.array, padding: int) -> np.array:
    print(f"{np.zeros((padding, a.shape[0])).shape=}")
    print(f"{a.transpose().shape=}")
    a_padded = np.vstack(
        [
            # Pad front - we'll be shifting by changing indices, instead of adding padding each iteration which would
            # cause re-allocs each time
            np.zeros(
                (padding, a.shape[0])
            ),
            # Transposing the spectrum matrix simplifies the padding logic
            a.transpose(),
        ]
    )

    print(f"{a_padded.shape=}")

    return a_padded.transpose()


def find_optimal_alignment(spectrum1: np.array, spectrum2: np.array) -> (int, float, float):
    original_length = spectrum1.shape[1]
    num_frequencies = spectrum1.shape[0]
    num_shifts = spectrum2.shape[1]

    spectrum1_padded = np.vstack(
        [
            # Pad front - we'll be shifting by changing indices, instead of adding padding each iteration which would
            # cause re-allocs each time
            np.zeros((num_shifts, num_frequencies)),
            # Transposing the spectrum matrix simplifies the padding logic
            spectrum1.transpose(),

        ]
    )

    # If spectrum1 is shorter in temporal length than spectrum2, add zero padding to the back since there is no sound
    # to compare.
    if num_shifts > original_length:
        spectrum1_padded = np.vstack(
            [
                spectrum1_padded,
                np.zeros((num_shifts - original_length, num_frequencies))
            ]
        )

    spectrum1_padded = np.ascontiguousarray(spectrum1_padded)
    spectrum2_t = np.ascontiguousarray(spectrum2.transpose())

    # means = np.array([float("inf") for _ in range(num_shifts)])
    # stds = np.array([float("inf") for _ in range(num_shifts)])
    #
    # for i in prange(num_shifts):
    #     # Truncate each row, so that spectrum 1's length does not exceed spectrum 2's length.
    #     start = num_shifts - i
    #
    #     # num_shifts is the length of spectrum2
    #     end = start + num_shifts
    #
    #     means[i], stds[i] = get_abs_diff(spectrum1_padded[start: end, :], spectrum2_t)
    #
    # spectrum1_best_shift = np.argmin(means)

    # return spectrum1_best_shift, means[spectrum1_best_shift], stds[spectrum1_best_shift]

    return _find_optimal_alignment(spectrum1_padded, spectrum2_t, num_shifts)


def align_sounds_by_power_spectra(spectrum1: np.array, spectrum2: np.array) -> (np.array, np.array):
    print("Testing spectrum1 alignment shifts")
    t = time.time()
    shift1, mean1, std1 = find_optimal_alignment(spectrum1, spectrum2)
    print(f"Took {time.time() - t} s")

    print("Testing spectrum2 alignment shifts")
    t = time.time()
    shift2, mean2, std2 = find_optimal_alignment(spectrum2, spectrum1)
    print(f"Took {time.time() - t} s")

    print(f"{shift1=} {mean1=} {std1=}")
    print(f"{shift2=} {mean2=} {std2=}")

    print(type(shift1))
    print(f"{spectrum1.shape=}")
    print(f"{spectrum2.shape=}")
    if mean1 < mean2:
        print(f"Shifting spectrum1 by {shift1=}")
        spectrum1 = pad_spectrum(spectrum1, shift1)

    elif mean2 < mean1:
        print(f"Shifting spectrum2 by {shift2=}")
        spectrum2 = pad_spectrum(spectrum2, shift2)

    # If the means are equal (unlikely), use whichever one has the smaller shift amount (because this means less will
    # be truncated)
    else:
        if shift1 < shift2:
            print(f"Shifting spectrum1 by {shift1=}")
            spectrum1 = pad_spectrum(spectrum1, shift1)

        else:
            print(f"Shifting spectrum2 by {shift2=}")
            spectrum2 = pad_spectrum(spectrum2, shift2)

    # Once the signals have been aligned, trim the excess from whichever signal is longer.
    if spectrum1.shape[1] > spectrum2.shape[1]:
        spectrum1 = spectrum1[:, :spectrum2.shape[1]]

    elif spectrum2.shape[1] > spectrum1.shape[1]:
        spectrum2 = spectrum2[:, :spectrum1.shape[1]]

    return spectrum1, spectrum2


def power_difference(sound1: np.array, sound2: np.array, sample_rate: int or float, start_frequency: int, end_frequency: int) -> (np.array, np.array, np.array):
    sound1_power = calculate_power_spectrum(sound1, sample_rate, start_frequency, end_frequency)
    sound2_power = calculate_power_spectrum(sound2, sample_rate, start_frequency, end_frequency)
    
    return sound1_power - sound2_power, sound1_power, sound2_power


def _register_balanced_frequency_scale():
    class BalancedFrequencyTransform(Transform):
        input_dims = output_dims = 1
    
        def transform_non_affine(self, frequencies):
            c = 400 / 44100
            return np.log(1 + frequencies * c) / np.log(1 + 400)
    
        def inverted(self):
            return BalancedFrequencyInverseTransform()

    class BalancedFrequencyInverseTransform(Transform):
        input_dims = output_dims = 1
    
        def transform_non_affine(self, x_norm):
            c = 400 / 44100
            return (np.exp(x_norm * np.log(1 + 400)) - 1) / c

    class BalancedFrequencyScale(ScaleBase):
        name = "balanced_frequency_scale"
    
        def get_transform(self):
            return BalancedFrequencyTransform()
    
        def set_default_locators_and_formatters(self, axis):
            axis.set_major_locator(plt.LogLocator())
            axis.set_major_formatter(plt.LogFormatter())

    matplotlib.scale.register_scale(BalancedFrequencyScale)


def plot_wavelet_power_spectrum(wavelet_power: np.array, sample_rate: int or float, start_frequency: int, end_frequency: int) -> None:
    print(f"{wavelet_power.mean()=}")
    print(f"{wavelet_power.std()=}")
    print(f"{wavelet_power.max()=}")
    print(f"{wavelet_power.min()=}")
    plt.figure(figsize=(24, 13.5), dpi=300)
    plt.imshow(
        wavelet_power,
        aspect="auto",
        extent=[0, wavelet_power.shape[1], end_frequency, start_frequency],
        cmap="seismic",
        norm=SymLogNorm(1, vmin=-10, vmax=10)
    )

    # Invert the y-axis so that low frequencies appear at the bottom.
    plt.gca().invert_yaxis()

    plt.yscale("balanced_frequency_scale")

    plt.tick_params(axis="both", which="major", labelsize=3)

    # TODO - this is hardcoded for F#2. Need a way to identify what note we're dealing with, then adding the ticks
    #   for that note and its harmonics.
    yticks = [i * 92.499 for i in range(int(7000 / 92.499))]
    # Drop 0, add subharmonics
    yticks = [92.499 / 4, 92.499 / 2] + yticks[1:]

    yticks = [float(f"{i:.2f}") for i in yticks]
    plt.yticks(yticks, [f"{i} Hz" for i in yticks])

    xticks = [i * 2000 for i in range(1, wavelet_power.shape[1] // 2000)]
    plt.xticks(xticks, [f"{i / sample_rate:.2f}" for i in xticks])

    # Add labels and titles.
    plt.title(f"Wavelet Power Spectrum", fontsize=6)
    plt.ylabel("Frequency (Hz)", fontsize=4)
    plt.xlabel("Time (s)", fontsize=4)
    plt.colorbar(label="Power")


_register_balanced_frequency_scale()
