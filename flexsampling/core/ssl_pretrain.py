"""Self-Supervised Contrastive Pre-training (Section 2.1).

Trains the encoder with contrastive learning to obtain category
distribution-agnostic features, avoiding semantic bias toward majority classes.

Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy)
  L = -log[ exp(sim(v_i, v_i+) / T) / sum_j exp(sim(v_i, v_j) / T) ]

where v_i+ is a positive pair (augmented view of the same image) and v_j
includes all other samples in the batch as negatives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Optional, Tuple


class ContrastiveAugmentation:
    """Produces two augmented views of the same image for contrastive learning."""

    def __init__(self, img_size: int = 224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class ContrastiveDataset(Dataset):
    """Wraps an existing dataset to produce paired augmented views.

    Replaces the dataset's transform with ContrastiveAugmentation so each
    __getitem__ returns ((view1, view2), label).
    """

    def __init__(self, base_dataset: Dataset, img_size: int = 224):
        self.base = base_dataset
        self.aug = ContrastiveAugmentation(img_size)
        # We need to access raw PIL images, so we'll reload from paths
        self._has_samples = hasattr(base_dataset, 'samples')
        if self._has_samples:
            self.samples = base_dataset.samples
            self._find_image = base_dataset._find_image

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if self._has_samples:
            from PIL import Image
            fname, label = self.samples[index]
            path = self._find_image(fname)
            img = Image.open(path).convert("RGB")
            view1, view2 = self.aug(img)
            return (view1, view2), label
        else:
            # Fallback: use the base dataset's image (already tensor)
            # and apply random transforms on top
            img, label = self.base[index]
            return (img, img), label


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent contrastive loss.

    Args:
        z1, z2: (N, D) L2-normalized projection vectors from two views.
        temperature: Scaling temperature T.
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = torch.mm(z, z.t()) / temperature  # (2N, 2N)

    # Mask out self-similarity
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    pos_i = torch.arange(N, device=z.device)
    labels = torch.cat([pos_i + N, pos_i], dim=0)  # (2N,)

    return F.cross_entropy(sim, labels)


class SSLPretrainer:
    """Self-supervised contrastive pre-training for the encoder.

    Args:
        encoder: Backbone network (outputs features, no classification head).
        embed_dim: Encoder output dimension.
        dataset: Training dataset (ISICDataset or similar).
        device: Torch device.
        img_size: Image size for augmentations.
        epochs: Number of SSL pre-training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        temperature: NT-Xent temperature.
        num_workers: DataLoader workers.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        dataset: Dataset,
        device: torch.device,
        img_size: int = 224,
        epochs: int = 100,
        batch_size: int = 256,
        accum_steps: int = 1,
        lr: float = 0.001,
        temperature: float = 0.5,
        num_workers: int = 0,
    ):
        self.encoder = encoder
        self.projector = ProjectionHead(embed_dim).to(device)
        self.device = device
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.accum_steps = int(accum_steps)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.num_workers = int(num_workers)

        self.contrastive_ds = ContrastiveDataset(dataset, img_size)

    def train(self) -> nn.Module:
        """Run contrastive pre-training and return the trained encoder."""
        loader = DataLoader(
            self.contrastive_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * len(loader),
        )

        self.encoder.train()
        self.projector.train()

        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            optimizer.zero_grad()
            for step, ((view1, view2), _) in enumerate(loader):
                view1 = view1.to(self.device, non_blocking=True)
                view2 = view2.to(self.device, non_blocking=True)

                # Forward through encoder + projector
                h1 = self.encoder(view1)
                h2 = self.encoder(view2)
                if h1.dim() > 2:
                    h1 = h1.mean(dim=list(range(1, h1.dim() - 1)))
                    h2 = h2.mean(dim=list(range(1, h2.dim() - 1)))

                z1 = F.normalize(self.projector(h1), dim=1)
                z2 = F.normalize(self.projector(h2), dim=1)

                loss = nt_xent_loss(z1, z2, self.temperature) / self.accum_steps
                loss.backward()

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.accum_steps
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [SSL] Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

        self.encoder.eval()
        return self.encoder
