import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNBlock(nn.Module):
    """A TDNN block with dilation."""
    def __init__(self, in_channels, out_channels, context):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=context,
            dilation=1,
            padding=context // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class XVectorNet(nn.Module):
    def __init__(self, num_speakers, embedding_dim=256):
        super().__init__()

        # TDNN layers
        self.tdnn1 = TDNNBlock(64, 128, context=5)
        self.tdnn2 = TDNNBlock(128, 128, context=3)
        self.tdnn3 = TDNNBlock(128, 128, context=3)

        # Statistics pooling
        # Converts variable-length T → fixed-size vector
        self.stat_pool = lambda x: torch.cat(
            (x.mean(dim=2), x.std(dim=2)), dim=1
        )

        # Fully-connected layers for embeddings
        self.fc1 = nn.Linear(128 * 2, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_speakers)

    def forward(self, x):
        # x: (batch, 64, T)

        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)

        # statistics pooling
        x = self.stat_pool(x)

        # speaker embedding
        emb = F.relu(self.fc1(x))

        # classification head
        out = self.fc2(emb)

        return out, emb  # return both for training + embedding extraction
