from projected_gradients.store import Store
import torch
from typing import List
from projected_gradients.projection import ProjectionStore, Projection
from projected_gradients.svd_projection import SVDProjectionStore
import numpy as np


class PerturbationStore(Store):
    def __init__(self, model, names_of_params: list[str] | str = "all", seed: int = 0):
        self.model = model
        if isinstance(names_of_params, str) and names_of_params == "all":
            names_of_params = [name for name, _ in model.named_parameters()]
        else:
            assert isinstance(names_of_params, list)
            assert all(isinstance(name, str) for name in names_of_params)
        super().__init__(names_of_params)
        self.rng = np.random.default_rng(seed)
        self.store = self.make_random_perturbation(names_of_params)

    def make_random_perturbation(self, names_of_params: List[str]) -> torch.Tensor:
        """
        Makes a random perturbation ea for the given parameter set
        """
        perturbations = {}
        all_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for name in names_of_params:
                param = all_params[name]
                # sample a random direction from the unit sphere
                perturbations[name] = (
                    torch.tensor(self.rng.normal(size=param.shape))
                    .to(param.device)
                    .type(param.dtype)
                )
                perturbations[name] /= torch.norm(perturbations[name])
        return perturbations

    def __mul__(self, scalar: float) -> "PerturbationStore":
        for name in self.names_of_params:
            self.store[name] *= scalar
        return self


def project_perturbations(
    perturbations: "PerturbationStore",
    projections: ProjectionStore,
) -> None:
    """
    Projects the perturbations away from the top singular vectors of the difference between the parameters.
    """
    for name, perturbation_vector in perturbations.store.items():
        try:
            p_map: Projection = projections.store[name]
        except KeyError:
            raise KeyError(f"Projection vector for {name} not found")

        with torch.no_grad():
            perturbations.store[name] = p_map.do_complementary_projection(
                perturbation_vector
            )


def make_projected_perturbations(
    sft_model,
    it_model,
    names_of_params: list[str] | str,
    ndim: int,
    seed: int = 0,
) -> tuple[PerturbationStore, ProjectionStore]:
    """
    Constructs a PerturbationStore object from the given parameters
    """

    perturbations = PerturbationStore(sft_model, names_of_params, seed)
    projections = SVDProjectionStore.make_projection_store(
        ndim=ndim,
        names_of_params=perturbations.names_of_params,
        sft_model=sft_model,
        it_model=it_model,
    )
    project_perturbations(perturbations, projections)

    return perturbations, projections
