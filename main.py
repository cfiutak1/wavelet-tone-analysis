import matplotlib.pyplot as plt
import time

import audio_processing
import wavelet

challenger_sample_rate, challenger_sound = audio_processing.load_wav("sound/F#2 (Processed) v5.wav")
original_sample_rate, original_sound = audio_processing.load_wav("sound/F#2 (Studio).wav")

if original_sample_rate != challenger_sample_rate:
    raise ValueError(f"The sample rates of the files must match ({original_sample_rate=}, {challenger_sample_rate=}")


challenger_sound, original_sound = audio_processing.prepare_sounds_for_comparison(
    challenger_sound,
    original_sound,
    original_sample_rate
)

audio_processing.save_wav(
    "challenger_trimmed.wav",
    challenger_sample_rate,
    challenger_sound
)

audio_processing.save_wav(
    "original_trimmed.wav",
    original_sample_rate,
    original_sound
)

start_frequency = 1
end_frequency = 7000

diff, sound1_power, sound2_power = wavelet.power_difference(
    challenger_sound,
    original_sound,
    original_sample_rate,
    start_frequency,
    end_frequency
)

wavelet.plot_wavelet_power_spectrum(
    diff,
    original_sample_rate,
    start_frequency,
    end_frequency
)

plt.savefig(f"diff-{int(time.time())}.png")
