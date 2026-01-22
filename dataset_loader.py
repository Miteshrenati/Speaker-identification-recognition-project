import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SpeakerDataset(Dataset):
    def __init__(self, feature_root, fixed_frames=200):
        self.feature_root = feature_root
        self.fixed_frames = fixed_frames
        self.samples = []
        self.speaker_to_id = {}

        # assign each speaker a label
        speakers = sorted(os.listdir(feature_root))
        for idx, spk in enumerate(speakers):
            self.speaker_to_id[spk] = idx
            spk_path = os.path.join(feature_root, spk)

            for file in os.listdir(spk_path):
                if file.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(spk_path, file), idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        feat = np.load(file_path)  # shape: (64, T)

        # crop or pad to fixed length
        if feat.shape[1] >= self.fixed_frames:
            start = np.random.randint(0, feat.shape[1] - self.fixed_frames + 1)
            feat = feat[:, start:start + self.fixed_frames]
        else:
            pad_width = self.fixed_frames - feat.shape[1]
            feat = np.pad(feat, ((0, 0), (0, pad_width)))

        feat = torch.tensor(feat, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return feat, label
