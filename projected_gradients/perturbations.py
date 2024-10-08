import gc
from typing import List, Literal

import numpy as np
import torch

from projected_gradients.ProjectionStore import ProjectionStore
from projected_gradients.projection import Projection
from projected_gradients.store import Store
from projected_gradients.svd_projection import SVDProjectionStore


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
                # rescale the perturbation to have the same norm as the parameter
                perturbations[name] /= torch.norm(perturbations[name])
                perturbations[name] *= torch.norm(param)
        return perturbations

    def __mul__(self, scalar: float) -> "PerturbationStore":
        """In place multiplication of the perturbations"""
        for name in self.names_of_params:
            self.store[name] *= scalar
        return self
    
    def dump(self, dump_dir: str) -> None:
        """
        Dumps the perturbations to the given directory
        """
        for name, perturbation in self.store.items():
            torch.save(perturbation, f"{dump_dir}/{name}.pt")
    
    def load(self, dump_dir: str) -> None:
        """
        Loads the perturbations from the given directory
        """
        self.store = {}
        gc.collect()
        torch.cuda.empty_cache()
        for name in self.names_of_params:
            self.store[name] = torch.load(f"{dump_dir}/{name}.pt")


def project_perturbations(
    perturbations: "PerturbationStore",
    projections: ProjectionStore,
) -> None:
    """
    Projects the perturbations away from the top singular vectors of the difference between the parameters.
    """
    for name, perturbation_vector in perturbations.store.items():
        gc.collect()
        torch.cuda.empty_cache()
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
    projection_type: Literal["left", "right", "both"] = "right",
    return_projections: bool = False,
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
        projection_type=projection_type,
    )
    project_perturbations(perturbations, projections)
    # check if NaNs are present
    assert not any(
        torch.isnan(perturbation).any() for perturbation in perturbations.store.values()
    ), "NaNs found in perturbations"
    if return_projections:
        return perturbations, projections
    del projections
    torch.cuda.empty_cache()
    return perturbations
