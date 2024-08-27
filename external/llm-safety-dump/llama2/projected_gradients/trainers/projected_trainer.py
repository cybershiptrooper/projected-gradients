from torch.nn.modules import Module

from projected_gradients.ProjectionStore import ProjectionStore
from .trainer_config import TrainerConfig
from .generic_trainer import GenericLLMTrainer
from projected_gradients.projection import Projection


class ProjectedTrainer(GenericLLMTrainer):
    def __init__(
        self, config: TrainerConfig, model: Module, projection_store: ProjectionStore
    ):
        super().__init__(config, model)
        self.projection_store = projection_store

    def run_train_step(self, batch) -> dict[str, float]:
        tokens = batch
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        # project the gradients
        for name, param in self.model.named_parameters():
            if name in self.projection_store:
                projection: Projection = self.projection_store[name]
                param.grad = projection.do_complementary_projection(param.grad)

        self.optimizer.step()
        return {"train/loss": loss.item()}
