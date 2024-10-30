import matplotlib.pyplot as plt
import time
import numpy as np

import audio_processing
import wavelet


challenger_sample_rate, challenger_sound = audio_processing.load_wav("challenger_trimmed.wav")
original_sample_rate, original_sound = audio_processing.load_wav("original_trimmed.wav")

if original_sample_rate != challenger_sample_rate:
    raise ValueError(f"The sample rates of the files must match ({original_sample_rate=}, {challenger_sample_rate=}")


if len(challenger_sound) < len(original_sound):
    print(f"Warning: Length of the audio files does not match, trimming the longer one. {len(challenger_sound)=} < {len(original_sound)=}")
    original_sound = original_sound[:len(challenger_sound)]


elif len(challenger_sound) > len(original_sound):
    print(f"Warning: Length of the audio files does not match, trimming the longer one. {len(challenger_sound)=} > {len(original_sound)=}")
    challenger_sound = challenger_sound[:len(original_sound)]


start_frequency = 1
end_frequency = 7000

diff, sound1_power, sound2_power = wavelet.power_difference(
    challenger_sound,
    original_sound,
    original_sample_rate,
    start_frequency,
    end_frequency
)

print(
    f"{((np.abs(sound1_power) < 0.0001) & (np.abs(sound2_power) < 0.0001)).sum()=}"
)

print(
    f"{sound1_power.shape=}"
)
print(
    f"{sound2_power.shape=}"
)

challenger_power, original_power = wavelet.align_sounds_by_power_spectra(sound1_power, sound2_power)
print(
    f"{challenger_power.shape=}"
)
print(
    f"{original_power.shape=}"
)



diff = challenger_power - original_power

wavelet.plot_wavelet_power_spectrum(
    diff,
    original_sample_rate,
    start_frequency,
    end_frequency
)



plt.savefig(f"diff-{int(time.time())}.png")
