# scripts/train_patchgan.py
"""
Monte Carlo Batch Sampling Trainer for PatchGAN Discriminator

Implements variance-reduced gradient estimation via Monte Carlo sampling of mini-batches.
Accumulates gradients over multiple independent forward passes before each optimizer step.

Key features:
- LSGAN-style training with MSE loss + label smoothing
- Monte Carlo sampling (multiple batches per update) → lower gradient variance
- Early stopping based on validation loss
- Checkpointing of best model
- Progress bar with real-time loss/variance stats
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import logging
from typing import Tuple, Optional, Dict, Any

from models.discriminator import PatchGANDiscriminator


class MonteCarloTrainer:
    """
    Trainer using Monte Carlo sampling to reduce gradient variance in PatchGAN training.
    
    Instead of one large batch per update, we sample many smaller batches (MC samples),
    accumulate gradients, and perform one optimizer step → more stable training.
    """

    def __init__(
        self,
        discriminator: PatchGANDiscriminator,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        mc_samples: int = 64,               # typical range: 32–128
        batch_size: int = 8,
        learning_rate: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            discriminator: Initialized PatchGANDiscriminator model
            train_dataset: Dataset returning (real_img, fake_img) pairs
            val_dataset: Optional validation dataset
            mc_samples: Number of mini-batches to sample per parameter update
            batch_size: Size of each mini-batch
            learning_rate: Adam learning rate
            beta1, beta2: Adam betas (0.5/0.999 common for GANs)
            device: 'cuda' or 'cpu'
        """
        self.discriminator = discriminator.to(device)
        self.device = device
        self.mc_samples = mc_samples
        self.batch_size = batch_size

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if device.startswith('cuda') else False,
            drop_last=True
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if device.startswith('cuda') else False
            )

        # Optimizer & loss
        self.optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2)
        )
        self.criterion = nn.MSELoss()

        # Logging & history
        self.logger = logging.getLogger('MonteCarloTrainer')
        self.train_history: Dict[str, list] = {'d_loss': [], 'variance': [], 'val_loss': []}

    def _get_labels(self, pred_shape: torch.Size, real: bool) -> torch.Tensor:
        """Generate target labels with one-sided label smoothing for real samples."""
        if real:
            return torch.full(pred_shape, 0.9, device=self.device, dtype=torch.float32)
        return torch.zeros(pred_shape, device=self.device, dtype=torch.float32)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch using Monte Carlo sampling.

        Returns:
            (mean_loss, mean_variance) over the epoch
        """
        self.discriminator.train()
        epoch_losses = []
        epoch_variances = []

        # Cycle through the dataset if needed
        data_iter = iter(self.train_loader)

        # Number of parameter updates per epoch
        num_updates = max(1, len(self.train_loader) // self.mc_samples)

        pbar = tqdm(range(num_updates), desc=f"Epoch {epoch:3d}")

        for _ in pbar:
            self.optimizer.zero_grad()
            mc_losses = []

            for _ in range(self.mc_samples):
                try:
                    real_imgs, fake_imgs = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    real_imgs, fake_imgs = next(data_iter)

                real_imgs = real_imgs.to(self.device)
                fake_imgs = fake_imgs.to(self.device)

                # Real pass
                real_pred = self.discriminator(real_imgs)
                real_target = self._get_labels(real_pred.shape, real=True)
                loss_real = self.criterion(real_pred, real_target)

                # Fake pass
                fake_pred = self.discriminator(fake_imgs)
                fake_target = self._get_labels(fake_pred.shape, real=False)
                loss_fake = self.criterion(fake_pred, fake_target)

                loss = (loss_real + loss_fake) / 2
                mc_losses.append(loss.item())

                # Accumulate scaled gradient
                (loss / self.mc_samples).backward()

            # One optimizer step after MC accumulation
            self.optimizer.step()

            mean_loss = np.mean(mc_losses)
            var_loss = np.var(mc_losses)
            epoch_losses.append(mean_loss)
            epoch_variances.append(var_loss)

            pbar.set_postfix(loss=f"{mean_loss:.4f}", var=f"{var_loss:.4f}")

        epoch_mean_loss = np.mean(epoch_losses)
        epoch_mean_var = np.mean(epoch_variances)

        self.train_history['d_loss'].append(epoch_mean_loss)
        self.train_history['variance'].append(epoch_mean_var)

        self.logger.info(
            f"Epoch {epoch:3d} | Train Loss: {epoch_mean_loss:.4f} ± {epoch_mean_var:.4f}"
        )

        return epoch_mean_loss, epoch_mean_var

    @torch.no_grad()
    def validate(self) -> float:
        """Compute average validation loss."""
        if self.val_loader is None:
            return 0.0

        self.discriminator.eval()
        val_losses = []

        for real_imgs, fake_imgs in self.val_loader:
            real_imgs = real_imgs.to(self.device)
            fake_imgs = fake_imgs.to(self.device)

            real_pred = self.discriminator(real_imgs)
            fake_pred = self.discriminator(fake_imgs)

            real_target = self._get_labels(real_pred.shape, True)
            fake_target = self._get_labels(fake_pred.shape, False)

            loss = (self.criterion(real_pred, real_target) +
                    self.criterion(fake_pred, fake_target)) / 2

            val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        self.train_history['val_loss'].append(val_loss)
        return val_loss

    def train(
        self,
        num_epochs: int = 200,
        save_path: str = "weights/patchgan_best.pt",
        early_stopping_patience: int = 25
    ) -> Dict[str, list]:
        """
        Full training loop with validation and early stopping.

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_var = self.train_epoch(epoch)
            val_loss = self.validate()

            self.logger.info(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            )

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.discriminator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'mc_samples': self.mc_samples,
                    'history': self.train_history
                }, save_path)

                self.logger.info(f"→ Saved best model at {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

        return self.train_history


# ── Example usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example instantiation (datasets must be provided by user)
    # disc = PatchGANDiscriminator(input_channels=3, ndf=64)
    # trainer = MonteCarloTrainer(
    #     discriminator=disc,
    #     train_dataset=YourPairedDataset(...),
    #     val_dataset=YourPairedDataset(...),
    #     mc_samples=64,
    #     batch_size=12
    # )
    # history = trainer.train(num_epochs=300, save_path="patchgan_mc.pt")
