import os
import numpy as np
import torch
import librosa
from model import XVectorNet
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIR = "features"              # extracted log-mel features folder
MODEL_PATH = "speaker_xvector_model.pth"


# -------------------- Load Model --------------------
def load_model(num_speakers):
    model = XVectorNet(num_speakers=num_speakers)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# -------------------- Cosine Similarity --------------------
def normalize(vector):
    return vector / (np.linalg.norm(vector) + 1e-8)


# -------------------- Build Speaker Embeddings --------------------
def build_speaker_database():
    speakers = sorted(os.listdir(FEATURE_DIR))
    num_speakers = len(speakers)

    print(f"Found {num_speakers} speakers. Building database...")

    model = load_model(num_speakers)

    speaker_db = {}

    for idx, spk in enumerate(speakers):
        print(f"\nProcessing speaker: {spk}")

        spk_path = os.path.join(FEATURE_DIR, spk)
        embeddings = []

        for file in os.listdir(spk_path):
            if file.endswith(".npy"):
                feat = np.load(os.path.join(spk_path, file))  # (64, T)

                feat_tensor = torch.tensor(feat).unsqueeze(0).to(DEVICE)  # (1, 64, T)
                _, emb = model(feat_tensor)

                emb = emb.detach().cpu().numpy()[0]
                embeddings.append(emb)

        # Average embedding for the speaker
        mean_emb = np.mean(embeddings, axis=0)
        speaker_db[spk] = normalize(mean_emb)

        print(f" → Saved embedding for {spk}, based on {len(embeddings)} samples.")

    # Save the entire database
    np.save("speaker_embeddings.npy", speaker_db)
    print("\n✅ Speaker embeddings saved to speaker_embeddings.npy")


# -------------------- RUN --------------------
if __name__ == "__main__":
    build_speaker_database()
 
 
 