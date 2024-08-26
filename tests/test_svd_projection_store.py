import os

import torch

from projected_gradients.svd_projection import SVDProjectionStore


def test_svd_projection_store():
    sft_model = torch.nn.Linear(3, 3)
    it_model = torch.nn.Linear(3, 3)
    sft_model.weight = torch.nn.Parameter(torch.ones(3, 3))
    it_model.weight = torch.nn.Parameter(torch.eye(3, 3))
    projection_store = SVDProjectionStore.make_projection_store(
        3, ["weight"], sft_model, it_model
    )
    assert projection_store.ndim == 3
    assert projection_store.names_of_params == ["weight"]

    top_singular_vector = projection_store["weight"].right_param[0]
    assert torch.allclose(top_singular_vector * 3**0.5, -torch.ones(3), atol=1e-3), (
        top_singular_vector * 3**0.5
    )


def test_keep_in_files():
    sft_model = torch.nn.Linear(3, 3)
    it_model = torch.nn.Linear(3, 3)
    root_dir = "results/test_model"
    os.makedirs(root_dir, exist_ok=True)
    projection_store = SVDProjectionStore.make_projection_store(
        3,
        ["weight", "bias"],
        sft_model,
        it_model,
        keep_in_files=True,
        dump_root_dir=root_dir,
    )

    assert os.path.exists(projection_store["weight"].right_param_file)
