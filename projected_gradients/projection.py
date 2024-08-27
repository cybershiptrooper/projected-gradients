from typing import Optional

import torch
from jaxtyping import Float


class Projection:
    def __init__(
        self,
        left_param: Optional[Float[torch.Tensor, "ndim d"]],  # noqa: F722
        right_param: Optional[Float[torch.Tensor, "ndim d"]],  # noqa: F722
        keep_in_files: bool = False,
        dump_dir: str | None = None,
    ):
        assert left_param is not None or right_param is not None, ValueError(
            "At least one of left_param or right_param must be provided"
        )
        if keep_in_files:
            assert dump_dir is not None, ValueError(
                "dump_file must be provided if keep_in_files is True"
            )
            self.left_param_file = f"{dump_dir}/left_param.pt"
            self.right_param_file = f"{dump_dir}/right_param.pt"
            if left_param is not None:
                torch.save(left_param, self.left_param_file)
            if right_param is not None:
                torch.save(right_param, self.right_param_file)
        else:
            self._left_param = left_param.to("cpu") if left_param is not None else None
            self._right_param = right_param.to("cpu") if right_param is not None else None

    @property
    def left_param(self) -> Optional[Float[torch.Tensor, "ndim d"]]:  # noqa: F722
        if hasattr(self, "left_param_file"):
            try:
                return torch.load(self.left_param_file)
            except FileNotFoundError:
                return None
        return self._left_param

    @property
    def right_param(self) -> Optional[Float[torch.Tensor, "ndim d"]]:  # noqa: F722
        if hasattr(self, "right_param_file"):
            try:
                return torch.load(self.right_param_file)
            except FileNotFoundError:
                return None
        return self._right_param

    def do_complementary_projection(self, perturbation: torch.tensor) -> torch.tensor:
        if self.left_param is not None:
            left_param = self.left_param.to(perturbation.device).type(
                perturbation.dtype
            )
            left_param_outer_products = torch.einsum(
                "ki,kj->ij", left_param, left_param
            )

            left_projection_matrix = torch.eye(left_param_outer_products.shape[-1]).to(
                perturbation.device
            ) - left_param_outer_products.to(perturbation.device)
        else:
            left_projection_matrix = torch.eye(perturbation.shape[0]).to(
                perturbation.device
            )

        if self.right_param is not None:
            right_param = self.right_param.to(perturbation.device).type(
                perturbation.dtype
            )
            right_param_outer_products = torch.einsum(
                "ki,kj->ij", right_param, right_param
            )

            right_projection_matrix = torch.eye(
                right_param_outer_products.shape[-1]
            ).to(perturbation.device) - right_param_outer_products.to(
                perturbation.device
            )
        else:
            right_projection_matrix = torch.eye(perturbation.shape[-1]).to(
                perturbation.device
            )
        left_projection_matrix = left_projection_matrix.type(perturbation.dtype)
        right_projection_matrix = right_projection_matrix.type(perturbation.dtype)
        ans = left_projection_matrix @ perturbation @ right_projection_matrix

        # save memory
        del left_projection_matrix, right_projection_matrix, left_param, right_param
        torch.cuda.empty_cache()

        return (ans / torch.norm(ans)) * torch.norm(perturbation)


class BiasProjection(Projection):
    def __init__(
        self,
        param: Float[torch.Tensor, "ndim d"],  # noqa: F722
        save_in_files: bool = False,
        dump_dir: str | None = None,
    ):
        super().__init__(None, param, save_in_files, dump_dir)

    def do_complementary_projection(self, perturbation: torch.tensor) -> torch.tensor:
        # biases projection is just removing the component
        # of the perturbation vector in the direction of the bias
        param = self.right_param.to(perturbation.device)
        component_along_bias = torch.einsum("i,i->", perturbation, param)
        ans = perturbation - component_along_bias * param
        return (ans / torch.norm(ans)) * torch.norm(perturbation)
