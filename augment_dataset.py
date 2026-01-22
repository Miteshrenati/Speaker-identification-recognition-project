import os
import numpy as np
import librosa
import soundfile as sf
import random

# ======================
# CONFIG
# ======================
INPUT_ROOT = r"F:\50_speakers_audio_data"
OUTPUT_ROOT = r"F:\augmented_clips"

SAMPLE_RATE = 16000
NARROW_RATE = 8000

PITCH_RANGE = 2           # semitones
SPEED_MIN = 0.90
SPEED_MAX = 1.10

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ======================
# HELPERS
# ======================

def trim_silence(y):
    yt, _ = librosa.effects.trim(y, top_db=20)
    return yt

def add_random_noise(y):
    """Add white, pink, or brown noise randomly."""
    choice = random.choice(["white", "pink", "brown"])

    if choice == "white":
        noise = np.random.randn(len(y))

    elif choice == "pink":
        uneven = len(y) % 2
        X = np.random.randn(len(y)//2 + 1 + uneven) + 1j*np.random.randn(len(y)//2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y_noise = (np.fft.irfft(X / S)).real
        noise = y_noise[:len(y)]

    else:  # brown noise
        noise = np.cumsum(np.random.randn(len(y)))
        noise = noise / np.max(np.abs(noise))

    noise_level = random.uniform(0.01, 0.05)
    noisy = y + noise * noise_level

    return np.clip(noisy, -1.0, 1.0)

def save(path, y, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)


# ======================
# MAIN PROCESS LOOP
# ======================
for speaker in os.listdir(INPUT_ROOT):
    spk_path = os.path.join(INPUT_ROOT, speaker)
    if not os.path.isdir(spk_path):
        continue

    print(f"\nProcessing speaker: {speaker}")

    out_spk = os.path.join(OUTPUT_ROOT, speaker)
    os.makedirs(out_spk, exist_ok=True)

    for file in os.listdir(spk_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(spk_path, file)
        name = os.path.splitext(file)[0]

        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # 1️⃣ Clean
        y_clean = trim_silence(y)
        save(os.path.join(out_spk, f"{name}_clean.wav"), y_clean, SAMPLE_RATE)

        # 2️⃣ Narrowband
        y_narrow = librosa.resample(y_clean, orig_sr=SAMPLE_RATE, target_sr=NARROW_RATE)
        save(os.path.join(out_spk, f"{name}_narrow.wav"), y_narrow, NARROW_RATE)

        # 3️⃣ Pitch shift
        n_steps = random.uniform(-PITCH_RANGE, PITCH_RANGE)
        y_pitch = librosa.effects.pitch_shift(y_clean, sr=SAMPLE_RATE, n_steps=n_steps)

        # 4️⃣ Speed change
        speed = random.uniform(SPEED_MIN, SPEED_MAX)
        y_pitchspeed = librosa.effects.time_stretch(y_pitch, rate=speed)
        save(os.path.join(out_spk, f"{name}_pitchspeed.wav"), y_pitchspeed, SAMPLE_RATE)

        # 5️⃣ Add random noise (no folder needed)
        y_noisy = add_random_noise(y_clean)
        save(os.path.join(out_spk, f"{name}_noisy.wav"), y_noisy, SAMPLE_RATE)

        print(f"  ✔ {file} → clean, narrow, pitchspeed, noisy")

