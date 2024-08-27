import os
from abc import ABC
from typing import Literal

import torch

from projected_gradients.projection import Projection
from projected_gradients.store import Store


class ProjectionStore(Store[str, Projection], ABC):
    def __init__(
        self,
        ndim: int,
        names_of_params: list[str],
        projection_type: Literal["left", "right", "both"] = "right",
        keep_in_files: bool = False,
        dump_root_dir: str | None = None,
    ):
        """
        Makes a projection store object with ndim vectors for the given parameter set
        """
        self.ndim = ndim
        self.projection_type = projection_type
        assert self.projection_type in ["left", "right", "both"], ValueError(
            "projection_type must be one of 'left', 'right', or 'both'"
        )
        self.keep_in_files = keep_in_files
        self.dump_root_dir = dump_root_dir

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
        projection_type: Literal["left", "right", "both"] = "right",
        **kwargs,
    ) -> "ProjectionStore":
        """
        Constructs a ProjectionStore object from the given parameters
        """
        projection_store = cls(ndim, names_of_params, projection_type, **kwargs)
        projection_store.store = projection_store.construct_projections(
            sft_model, it_model
        )
        return projection_store

    def make_param_dir(self, name_of_param: str) -> str:
        if self.dump_root_dir is None:
            return None
        param_dir = os.path.join(self.dump_root_dir, name_of_param)
        os.makedirs(param_dir, exist_ok=True)
        return param_dir
