import torch

from projected_gradients.projection import Projection, ProjectionStore


class SVDProjectionStore(ProjectionStore):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def construct_projections(
        self, sft_model: torch.nn.Module, it_model: torch.nn.Module
    ) -> dict:
        """
        Constructs the projections for the given models
        """
        projections = {}
        self.projection_values = {}
        for name in self.names_of_params:
            if "bias" in name:
                # TODO: Implement bias projection
                raise NotImplementedError("Bias projection not implemented")

            param = sft_model.state_dict()[name]
            corresponding_param = it_model.state_dict()[name]
            projections[name], self.projection_values[name] = self._svd_projection(
                param, corresponding_param
            )

        return projections

    def bias_diff(self, sft_bias: torch.Tensor, it_bias: torch.Tensor) -> torch.Tensor:
        """
        Returns the directional change in bias.
        """
        change_in_bias = sft_bias - it_bias
        change_in_bias /= torch.norm(change_in_bias)

        return change_in_bias


    def _svd_projection(
        self, sft_param: torch.Tensor, it_param: torch.Tensor
    ) -> tuple[Projection, torch.Tensor]:
        """
        Returns the top ndim right singular vectors of the difference between the parameters.
        """
        with torch.no_grad():
            param_diff = sft_param - it_param
            u, s, v = torch.svd(param_diff)
            v = v.transpose(0, -1)
            # pick first ndim columns of v and u
            top_right_singular_vectors = v[:, : self.ndim].T
            top_left_singular_vectors = u[:, : self.ndim].T

        projection = Projection(
            left_param=top_left_singular_vectors
            if self.projection_type in ["left", "both"]
            else None,
            right_param=top_right_singular_vectors
            if self.projection_type in ["right", "both"]
            else None,
        )

        return projection, param_diff
