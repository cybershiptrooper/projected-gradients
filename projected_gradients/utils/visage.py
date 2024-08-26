import copy
import gc
from typing import Literal

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projected_gradients.perturbations import (
    PerturbationStore,
    make_projected_perturbations,
    project_perturbations,
)
from projected_gradients.utils import make_safety_score
from projected_gradients.utils.tqdm import tqdm
from projected_gradients.ProjectionStore import ProjectionStore


def make_perturbed_model(
    model: AutoModelForCausalLM,
    perturbation: PerturbationStore,
) -> AutoModelForCausalLM:
    """
    Pertubs the model by adding a perturbation to the model's weights.
    """
    # make a copy of the model
    perturbed_model = copy.deepcopy(model)

    # add the perturbation to the model's weights
    for name, perturbation_vector in perturbation.store.items():
        perturbed_model.state_dict()[name] += perturbation_vector

    return perturbed_model


def visage(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    detect_toks: list[int],
    ndim: int | None = None,
    names_of_params: list[str] | str = "all",
    ratios: list = np.linspace(-5, 5, 11),
    safety_score_type: str = Literal["log_odds", "substring"],
    projected: bool = False,
    score_fn_kwargs: dict = {},
    it_model: AutoModelForCausalLM | None = None,
    projections: ProjectionStore | None = None,
    projection_type: Literal["right", "left", "both"] = "right",
    seed: int = 0,
):
    if projected:
        assert it_model is not None or projections is not None, ValueError(
            "If projected is True, either it_model or projections must be provided."
        )
        if it_model is not None:
            assert ndim is not None, ValueError(
                "If no projections are provided, ndim must be provided."
            )
            perturbation = make_projected_perturbations(
                model, it_model, names_of_params, ndim, seed, projection_type
            )
        else:
            perturbation = PerturbationStore(model, names_of_params, seed)
            project_perturbations(perturbation, projections)
    else:
        perturbation = PerturbationStore(model, names_of_params, seed)
    pbar = tqdm(total=len(ratios), desc="Generating safety scores")
    safety_scores = np.zeros_like(ratios)

    for i, r in enumerate(ratios):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pbar.update(1)
        perturbed_model = make_perturbed_model(model, perturbation * r)
        safety_score_fn = make_safety_score(
            safety_score_type, perturbed_model, tokenizer
        )
        safety_scores[i] = safety_score_fn(
            prompt=prompts, detect_toks=detect_toks, **score_fn_kwargs
        )
        del perturbed_model

    pbar.close()
    return safety_scores
