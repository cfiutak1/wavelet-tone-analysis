import audio_processing


challenger_sample_rate, challenger_sound = audio_processing.load_wav("sound/F#2 (Processed) v54.wav")
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
