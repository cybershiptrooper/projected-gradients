import torch

from projected_gradients.ProjectionStore import ProjectionStore
from projected_gradients.projection import Projection, BiasProjection


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
            param = sft_model.state_dict()[name]
            corresponding_param = it_model.state_dict()[name]
            call_method = self._svd_projection
            if "bias" in name:
                call_method = self.bias_diff
            projections[name], self.projection_values[name] = call_method(
                param, corresponding_param, name_of_param=name
            )

        return projections

    def bias_diff(
        self,
        sft_bias: torch.Tensor,
        it_bias: torch.Tensor,
        name_of_param: str | None = None,
    ) -> tuple[BiasProjection, None]:
        """
        Returns the directional change in bias.
        """
        change_in_bias = sft_bias - it_bias
        change_in_bias /= torch.norm(change_in_bias)

        return BiasProjection(
            change_in_bias,
            self.keep_in_files,
            self.make_param_dir(name_of_param=name_of_param),
        ), None

    def _svd_projection(
        self,
        sft_param: torch.Tensor,
        it_param: torch.Tensor,
        name_of_param: str | None = None,
    ) -> tuple[Projection, torch.Tensor]:
        """
        Returns the top ndim right singular vectors of the difference between the parameters.
        """
        with torch.no_grad():
            param_diff = (sft_param - it_param).to(torch.float32)
            u, s, v = torch.linalg.svd(param_diff, full_matrices=True)
            # convert back to sft param's dtype
            u = u.to(sft_param.dtype)
            v = v.to(sft_param.dtype)
            s = s.to(sft_param.dtype)
            
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
            keep_in_files=self.keep_in_files,
            dump_dir=self.make_param_dir(name_of_param=name_of_param),
        )

        return projection, s[: self.ndim]
