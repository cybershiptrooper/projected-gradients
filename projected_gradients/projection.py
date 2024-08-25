from abc import ABC
from typing import Optional

import torch
from jaxtyping import Float

from projected_gradients.store import Store


class Projection:
    def __init__(
        self,
        left_param: Optional[Float[torch.Tensor, "ndim d"]],  # noqa: F722
        right_param: Optional[Float[torch.Tensor, "ndim d"]],  # noqa: F722
    ):
        assert left_param is not None or right_param is not None, ValueError(
            "At least one of left_param or right_param must be provided"
        )
        self.left_param = left_param
        self.right_param = right_param

    def do_complementary_projection(self, perturbation: torch.tensor) -> torch.tensor:
        left_param = self.left_param
        right_param = self.right_param

        if left_param is not None:
            left_param_outer_products = torch.einsum(
                "ki,kj->ij", left_param, left_param
            )
            left_projection_matrix = (
                torch.eye(left_param_outer_products.shape[-1])
                - left_param_outer_products
            )
        else:
            left_projection_matrix = torch.eye(perturbation.shape[-1])

        if right_param is not None:
            right_param_outer_products = torch.einsum(
                "ki,kj->ij", right_param, right_param
            )
            right_projection_matrix = (
                torch.eye(right_param_outer_products.shape[-1])
                - right_param_outer_products
            )
        else:
            right_projection_matrix = torch.eye(perturbation.shape[-1])

        ans = left_projection_matrix @ perturbation @ right_projection_matrix
        return ans / torch.norm(ans)


class ProjectionStore(Store[str, Projection], ABC):
    def __init__(self, ndim: int, names_of_params: list[str]):
        """
        Makes a projection store object with ndim vectors for the given parameter set
        """
        self.ndim = ndim
        super().__init__(names_of_params)

    def construct_projections(
        self, sft_model: torch.nn.Module, it_model: torch.nn.Module
    ) -> dict[str, Projection]:
        """
        Constructs the projections for the given models
        """
        raise NotImplementedError

    @classmethod
    def make_projection_store(
        cls,
        ndim: int,
        names_of_params: list[str],
        sft_model: torch.nn.Module,
        it_model: torch.nn.Module,
    ) -> "ProjectionStore":
        """
        Constructs a ProjectionStore object from the given parameters
        """
        projection_store = cls(ndim, names_of_params)
        projection_store.store = projection_store.construct_projections(
            sft_model, it_model
        )
        return projection_store
