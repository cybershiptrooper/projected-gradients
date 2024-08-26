import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader


class DatasetManager:
    """Dataset for next token prediction. The datasets are quite small for fine-tuning, so we load them all into memory."""

    def __init__(
        self,
        data: Float[Tensor, "batch seq"],  # noqa: F722
        seed: int = 0,
        split_ratio: float = 0.8,
    ):
        train_size = int(data.shape[0] * split_ratio)

        # split data into train and val
        self.train_data = data[:train_size]
        self.val_data = data[train_size:]
        self.seed = seed

    def make_loaders(self, batch_size, num_workers) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return train_loader, val_loader
