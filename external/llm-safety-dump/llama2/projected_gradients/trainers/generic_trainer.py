import torch
from jaxtyping import Float
from torch import Tensor

from .base_trainer import BaseTrainer
from .metrics import Metric, MetricStore, MetricType


class GenericLLMTrainer(BaseTrainer):
    """Trainer for next token prediction tasks."""

    @property
    def criterion(
        self,
        output: Float[Tensor, "batch seq d_vocab"],  # noqa
        target: Float[Tensor, "batch seq"],  # noqa
    ) -> Float[Tensor, "batch"]:  # noqa
        return torch.nn.CrossEntropyLoss()(
            output.view(-1, output.size(-1)), target.view(-1)
        )

    def make_train_metrics(self) -> MetricStore:
        return MetricStore(
            [
                Metric("train/loss", MetricType.LOSS),
            ]
        )

    def make_eval_metrics(self) -> MetricStore:
        return MetricStore(
            [
                Metric("val/loss", MetricType.LOSS),
                Metric("val/accuracy", MetricType.ACCURACY),
            ]
        )

    def run_train_step(
        self,
        batch: Float[Tensor, "batch seq"],  # noqa: F722
    ) -> dict[str, float]:
        tokens = batch
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {"train/loss": loss.item()}

    def run_eval_step(
        self,
        batch: Float[Tensor, "batch seq"],  # noqa: F722
    ) -> dict[str, float]:
        tokens = batch
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(1) == targets).float().mean().item()
        return {"val/loss": loss.item(), "val/accuracy": acc}

    def check_early_stop_condition(self, eval_metrics: MetricStore) -> bool:
        return False
