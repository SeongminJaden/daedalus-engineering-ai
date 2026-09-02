"""A learned embedding of a part's surface, graded SURROGATE.

A small PointNet: a shared per-point network, a max over points, and a head
that produces a unit vector. The cloud arrives already centred, scaled and
rotated onto its principal axes (`pointcloud.canonical_frame`), and training
augments with the four proper sign flips that alignment leaves undecided.
What comes out is a similarity space: parts of one family close together,
parts of different families apart.

WHAT WAS MEASURED, on 150 training and 50 held-out synthetic parts
==================================================================
    22 descriptors, nearest neighbour       precision at 1   1.00
    D2 distance histogram                                     0.64
    PointNet, free rotation augmentation, 300 epochs           0.86 to 0.92
    PointNet, aligned, sign flips, 150 epochs (4 s on GPU)     0.88

    same part, randomly rotated, cosine to itself:
        free rotation augmentation   min 0.19   median 0.86
        aligned first                min 1.00   median 1.00

So the learned embedding beats the histogram that needed no learning and does
not beat the descriptors that needed no learning. On five prismatic families
the topology is the answer, and a learned space has shown that it can recover
most of it from points alone. The place it would earn its keep is shapes the
descriptors do not separate, which this training set does not contain.

WHAT AN EMBEDDING IS AND IS NOT
===============================
It is a learned similarity, and nothing more. Two parts being close in it is
a SUGGESTION that they are alike, graded SURROGATE like every learned output
here, and it decides nothing. It is measured against two things that needed
no learning, the 22 descriptors and the D2 distance histogram, on the one
task the training set can pose: does the nearest neighbour of a held-out
part share its family. If the learned space is not better than the baselines
on that task, it has learned nothing the geometry did not already say, and
the number is reported either way.

The five synthetic families are all this has seen. Its opinion of a part
outside them is a distance to things it knows, and the tests measure how far
the Fusion fixtures land rather than assuming they land anywhere sensible.

Dependencies: torch, already present for the surrogate. No new package.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from brain.semantic.evidence import EvidenceKind, EvidenceLevel

from .schema import label

EMBEDDING_DIM = 32
POINTS_PER_PART = 512


class PointNetEncoder(nn.Module):
    """Per-point MLP, max pool, projection to a unit vector."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, width: int = 128):
        super().__init__()
        self.per_point = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, width), nn.ReLU(),
            nn.Linear(width, width * 2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2, width), nn.ReLU(),
            nn.Linear(width, embedding_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (batch, n, 3)
        features = self.per_point(points).max(dim=1).values
        return nn.functional.normalize(self.head(features), dim=1)


#: The four proper sign flips of a principal frame. An odd number of flips is
#: a reflection, which a rotation can never produce, so it is not here.
SIGN_FLIPS = np.array([np.diag(s) for s in
                       ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])],
                      dtype=np.float32)


def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniformly random rotation matrices, by QR of Gaussian matrices with
    the sign fixed so that the result is a proper rotation."""
    out = np.empty((n, 3, 3))
    for i in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        out[i] = q
    return out


@dataclass
class EmbeddingBundle:
    encoder: PointNetEncoder
    families: list[str]
    train_metrics: dict = field(default_factory=dict)
    device: str = "cpu"

    def embed(self, clouds: np.ndarray, batch: int = 64) -> np.ndarray:
        """Unit embeddings for (n, points, 3) normalised clouds."""
        self.encoder.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(clouds), batch):
                x = torch.as_tensor(clouds[start:start + batch],
                                    dtype=torch.float32, device=self.device)
                out.append(self.encoder(x).cpu().numpy())
        return np.concatenate(out, axis=0)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), directory / "encoder.pt")
        (directory / "meta.json").write_text(json.dumps({
            "families": self.families, "train_metrics": self.train_metrics,
            "embedding_dim": EMBEDDING_DIM,
            "evidence": EvidenceLevel.SURROGATE.value}, indent=2))
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None
             ) -> "EmbeddingBundle":
        from surrogate.models import resolve_device

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        dev = resolve_device(device)
        encoder = PointNetEncoder(meta["embedding_dim"])
        encoder.load_state_dict(torch.load(directory / "encoder.pt",
                                           map_location=dev))
        encoder.to(dev)
        return cls(encoder=encoder, families=meta["families"],
                   train_metrics=meta["train_metrics"], device=str(dev))


def train_embedding(clouds: np.ndarray, families: list[str],
                    epochs: int = 150, seed: int = 0, lr: float = 1e-3,
                    batch: int = 16, device: str | None = None
                    ) -> EmbeddingBundle:
    """Fit the encoder with a family classification head on top.

    The head is thrown away; the embedding underneath it is kept. Clouds must
    already be in their canonical frame. Every batch gets a random proper
    sign flip and a small jitter, so the space is asked to be blind to the one
    ambiguity alignment leaves rather than trusted to be.
    """
    from surrogate.models import resolve_device

    dev = resolve_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = sorted(set(families))
    index = {f: i for i, f in enumerate(names)}
    y = torch.as_tensor([index[f] for f in families], device=dev)
    x_all = torch.as_tensor(clouds, dtype=torch.float32, device=dev)

    encoder = PointNetEncoder().to(dev)
    head = nn.Linear(EMBEDDING_DIM, len(names)).to(dev)
    optimiser = torch.optim.Adam(list(encoder.parameters())
                                 + list(head.parameters()), lr=lr)
    started = time.perf_counter()
    losses = []
    n = len(clouds)
    for _ in range(epochs):
        encoder.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            flips = torch.as_tensor(SIGN_FLIPS[rng.integers(0, 4, len(idx))],
                                    device=dev)
            x = torch.bmm(x_all[idx], flips)
            x = x + 0.01 * torch.randn_like(x)
            logits = head(encoder(x))
            loss = nn.functional.cross_entropy(logits, y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(idx)
        losses.append(epoch_loss / n)
    bundle = EmbeddingBundle(encoder=encoder, families=names, device=str(dev))
    bundle.train_metrics = {"epochs": epochs, "final_loss": losses[-1],
                            "first_loss": losses[0], "n_train": n,
                            "seconds": time.perf_counter() - started,
                            "device": str(dev)}
    return bundle


# ------------------------------------------------------------- evaluation

def nearest_neighbour_precision(embeddings: np.ndarray, families: list[str],
                                reference: np.ndarray | None = None,
                                reference_families: list[str] | None = None
                                ) -> float:
    """Fraction of parts whose nearest neighbour shares their family.

    Leave-one-out within one set when no reference is given; otherwise each
    query's nearest neighbour in the reference set.
    """
    if reference is None:
        reference, reference_families = embeddings, families
        d = np.linalg.norm(embeddings[:, None, :] - reference[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
    else:
        d = np.linalg.norm(embeddings[:, None, :] - reference[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    hits = sum(reference_families[j] == f for j, f in zip(nearest, families))
    return hits / max(len(families), 1)


def embedding_label(vector: np.ndarray, method: str,
                    kind: EvidenceKind = EvidenceKind.SURROGATE) -> dict:
    """An embedding as a dataset label. Learned ones are SURROGATE; the D2
    signature is ANALYTICAL and grades SIMULATED like any computed label."""
    return label(len(vector), "dimensions", kind, method,
                 note="a similarity, not a measurement",
                 vector=[float(v) for v in vector])
