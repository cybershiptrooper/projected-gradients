import torch

from projected_gradients.perturbations import make_projected_perturbations


def test_make_projected_perturbations():
    sft_model = torch.nn.Linear(3, 3)
    it_model = torch.nn.Linear(3, 3)
    sft_model.weight = torch.nn.Parameter(torch.ones(3, 3))
    it_model.weight = torch.nn.Parameter(torch.eye(3, 3))
    names_of_params = ["weight"]
    k = 1
    perturbation, projection = make_projected_perturbations(
        sft_model, it_model, names_of_params, k, return_projections=True
    )

    assert perturbation.names_of_params == projection.names_of_params
    for name, perturbation_vector in perturbation.store.items():
        assert perturbation_vector.shape == torch.Size([3, 3])
        assert torch.allclose(torch.norm(perturbation_vector), torch.norm(sft_model.weight))

    with torch.no_grad():
        diff_matrix = sft_model.weight - it_model.weight
        u, s, v = torch.linalg.svd(diff_matrix, full_matrices=True)
        top_v_outer_products = torch.einsum("ki,kj->ij", v[:k], v[:k])
        diff_projected_to_top_singular_vector = diff_matrix @ top_v_outer_products
        diff_projected_to_top_singular_vector /= torch.norm(
            diff_projected_to_top_singular_vector
        )

    perturbation_vector = perturbation.store["weight"] / torch.norm(
        perturbation.store["weight"]
    )
    print(perturbation_vector @ top_v_outer_products)
    assert torch.isclose(
        perturbation_vector @ top_v_outer_products,
        torch.zeros_like(perturbation_vector),
        atol=1e-6,
    ).all()

    print(perturbation_vector @ diff_projected_to_top_singular_vector.T)
    assert torch.isclose(
        perturbation_vector @ diff_projected_to_top_singular_vector.T,
        torch.zeros_like(perturbation_vector),
        atol=1e-6,
    ).all()
