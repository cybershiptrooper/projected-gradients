from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class TrainerConfig:
    lr: float
    batch_size: int
    num_epochs: int
    optimizer: torch.optim.Optimizer
    dataset: torch.utils.data.Dataset

    # wandb
    wandb_logging_enabled: bool
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # extra utils
    num_workers: int = 0
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    max_grad_norm: Optional[float] = None
    weight_decay: float = 0.0
    early_stopping_patience: int = 0

    # misc
    verbose: bool = False

    def __post_init__(self):
        if self.max_grad_norm is not None:
            assert self.max_grad_norm > 0.0, "max_grad_norm should be greater than 0.0"
        assert self.lr > 0.0, "Learning rate should be greater than 0.0"
        assert self.batch_size > 0, "Batch size should be greater than 0"
        assert self.num_epochs > 0, "Number of epochs should be greater than 0"

        if self.wandb_logging_enabled:
            assert (
                self.wandb_project is not None
            ), "wandb_project should be provided if wandb_logging_enabled is True"
            assert (
                self.wandb_entity is not None
            ), "wandb_entity should be provided if wandb_logging_enabled is True"
            if self.wandb_run_name is None:
                self.wandb_run_name = f"{self.dataset.__class__.__name__}_lr_{self.lr}_bs_{self.batch_size}_epochs_{self.num_epochs}"
