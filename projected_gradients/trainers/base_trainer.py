import torch
from .trainer_config import TrainerConfig
from .metrics import MetricStore
from abc import ABC, abstractmethod
import wandb


class BaseTrainer(ABC):
    def __init__(self, config: TrainerConfig, model: torch.nn.Module):
        self.config = config
        self.model = model
        self.optimizer = config.optimizer(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    @abstractmethod
    def make_train_metrics(self) -> MetricStore:
        raise NotImplementedError

    @abstractmethod
    def make_eval_metrics(self) -> MetricStore:
        raise NotImplementedError

    @abstractmethod
    def run_train_step(self, batch) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def run_eval_step(self, batch) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def check_early_stop_condition(self, eval_metrics: MetricStore) -> bool:
        raise NotImplementedError

    def _run_train_epoch(self, train_loader) -> MetricStore:
        train_metrics = self.make_train_metrics()
        self.model.train()
        for batch in train_loader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics = self.run_train_step(batch)
            train_metrics.update(metrics)
        return train_metrics

    def _run_eval_epoch(self, eval_loader) -> MetricStore:
        eval_metrics = self.make_eval_metrics()
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                metrics = self.run_eval_step(batch)
                eval_metrics.update(metrics)
        return eval_metrics

    def train(self):
        train_loader, eval_loader = self.config.dataset.make_loaders(
            self.config.batch_size, num_workers=self.config.num_workers
        )
        if self.config.wandb_logging_enabled and wandb.run is None:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                entity=self.config.wandb_entity,
            )
        for epoch in range(self.config.num_epochs):
            train_metrics = self._run_train_epoch(train_loader)
            eval_metrics = self._run_eval_epoch(eval_loader)
            self._print_and_log_metrics(
                epoch, train_metrics + eval_metrics, self.config.use_wandb
            )
            if self.check_early_stop_condition(eval_metrics):
                print("Early stopping condition reached!")
                break
            if self.config.lr_scheduler is not None:
                self.config.lr_scheduler.step()
        if self.config.wandb_logging_enabled:
            wandb.finish()

    @staticmethod
    def _print_and_log_metrics(epoch, metrics: MetricStore, use_wandb: bool):
        print(f"\nEpoch {epoch}:", end=" ")
        if use_wandb:
            wandb.log({"epoch": epoch})
        for metric in metrics:
            print(metric, end=", ")
            if use_wandb:
                wandb.log({metric.get_name(): metric.get_value()})
        print()
