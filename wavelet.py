import numpy as np
import pycwt
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
import matplotlib.pyplot as plt
import matplotlib.scale
from matplotlib.colors import CenteredNorm, SymLogNorm


def calculate_power_spectrum(sound: np.array, sample_rate: int or float, start_frequency: int, end_frequency: int) -> np.array:
    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
        sound,
        dt=1 / sample_rate,
        wavelet=pycwt.Morlet(6),
        freqs=np.array([i for i in range(start_frequency, end_frequency)])
    )
    
    wave_power = np.abs(wave) ** 2
    
    return wave_power


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
    plt.figure(figsize=(24, 13.5), dpi=300)
    plt.imshow(
        wavelet_power,
        aspect="auto",
        extent=[0, wavelet_power.shape[1], end_frequency, start_frequency],
        cmap="seismic",
        norm=SymLogNorm(2)
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

    # TODO - this is hardcoded for the current audio sample lengths.
    xticks = [i * 2000 for i in range(1, 8)]
    plt.xticks(xticks, [f"{i / sample_rate:.2f}" for i in xticks])

    # Add labels and titles.
    plt.title(f"Wavelet Power Spectrum", fontsize=6)
    plt.ylabel("Frequency (Hz)", fontsize=4)
    plt.xlabel("Time (s)", fontsize=4)
    plt.colorbar(label="Power")


_register_balanced_frequency_scale()
