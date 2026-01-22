import os
import numpy as np
import librosa

# ==========================================================
# CONFIG
# ==========================================================
AUDIO_ROOT = r"F:\augmented_clips"       # your augmented dataset
FEATURE_ROOT = "features"                # saved in current folder
N_MELS = 64                              # best for speaker models

os.makedirs(FEATURE_ROOT, exist_ok=True)


# ==========================================================
# FUNCTION: EXTRACT LOG-MEL FEATURES
# ==========================================================
def extract_logmel(y, sr, n_mels=N_MELS):
    """
    Extract log-mel filterbanks with per-utterance normalization.
    Returns feature shape: (n_mels, T)
    """

    # Window = 25 ms, Hop = 10 ms
    win_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    # Correct FFT size (nearest power of 2 >= win_length)
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    # Mel-spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0
    )

    # Convert to log scale
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Mean-variance normalization (VERY important)
    mean = np.mean(log_mel, axis=1, keepdims=True)
    std = np.std(log_mel, axis=1, keepdims=True) + 1e-10
    log_mel_norm = (log_mel - mean) / std

    return log_mel_norm.astype(np.float32)


# ==========================================================
# MAIN LOOP: PROCESS SPEAKERS AND FILES
# ==========================================================
print("\n🔍 Starting Feature Extraction...\n")

for speaker in os.listdir(AUDIO_ROOT):
    spk_path = os.path.join(AUDIO_ROOT, speaker)
    if not os.path.isdir(spk_path):
        continue

    print(f"[Speaker] {speaker}")

    # Create output folder for this speaker
    out_spk = os.path.join(FEATURE_ROOT, speaker)
    os.makedirs(out_spk, exist_ok=True)

    for file in os.listdir(spk_path):
        if not file.endswith(".wav"):
            continue

        audio_path = os.path.join(spk_path, file)

        # Load audio with original sample rate (8k or 16k)
        y, sr = librosa.load(audio_path, sr=None)

        if len(y) == 0:
            print(f"  ⚠ Skipping empty file: {file}")
            continue

        # Extract log-mel features
        features = extract_logmel(y, sr)

        # Save as .npy file
        base = os.path.splitext(file)[0]
        feat_path = os.path.join(out_spk, base + ".npy")
        np.save(feat_path, features)

        print(f"  ✔ {file}  →  {features.shape}")

print("\n✅ Feature Extraction Completed Successfully!")
print(f"➡ Features saved in: {os.path.abspath(FEATURE_ROOT)}")
