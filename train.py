import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader import SpeakerDataset
from model import XVectorNet


# ---------------- CONFIG ----------------
FEATURE_DIR = "features"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# ---------------- LOAD DATA ----------------
dataset = SpeakerDataset(FEATURE_DIR, fixed_frames=200)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

num_speakers = len(dataset.speaker_to_id)
print(f"Detected {num_speakers} speakers.")


# ---------------- BUILD MODEL ----------------
model = XVectorNet(num_speakers=num_speakers, embedding_dim=256).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ---------------- TRAIN LOOP ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for feats, labels in loader:
        feats = feats.to(DEVICE)        # (B, 64, 200)
        labels = labels.to(DEVICE)      # (B)

        optimizer.zero_grad()

        outputs, embeddings = model(feats)  # model returns (logits, embedding)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")


# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "speaker_xvector_model.pth")
print("\n✅ Training complete! Model saved as speaker_xvector_model.pth")
