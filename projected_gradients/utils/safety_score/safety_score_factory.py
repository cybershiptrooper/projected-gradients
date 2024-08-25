from typing import List, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .safety_metrics import (
    _test_prefixes_jailbreakbench,
    log_odds_metric,
    substring_matching_judge_fn,
)


def make_safety_score(
    safety_score_name: Literal["log_odds", "substring"],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> "Score":
    if safety_score_name == "log_odds":
        return LogOddsScore(model, tokenizer)
    elif safety_score_name == "substring":
        return SubstringMatchingScore(model, tokenizer)
    else:
        raise ValueError(f"Unknown safety score name: {safety_score_name}")


class Score:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt: List[str], *args, **kwargs):
        return self.score_fn(prompt, *args, **kwargs)

    def score_fn(self, prompt, detect_toks, *args, **kwargs):
        raise NotImplementedError


class LogOddsScore(Score):
    def score_fn(self, prompt: List[str], detect_toks: List[int]):
        assert self.tokenizer.padding_side == "left", ValueError(
            "Tokenizer must pad to the left"
        )
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(prompt["input_ids"].to(self.model.device))[0]
        return log_odds_metric(logits, detect_toks).mean().item()


class SubstringMatchingScore(Score):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def score_fn(
        self,
        prompt: List[str],
        detect_toks: List[str] = _test_prefixes_jailbreakbench,
        sample_length: int = 20,
        **kwargs,
    ):
        assert self.tokenizer.padding_side == "left", ValueError(
            "Tokenizer must pad to the left"
        )
        prompt_tokenised = self.tokenizer(prompt, return_tensors="pt", padding=True)
        for k, v in prompt_tokenised.items():
            prompt_tokenised[k] = v.to(self.model.device)
        with torch.no_grad():
            completion = self.model.generate(
                **prompt_tokenised, max_length=sample_length, **kwargs
            )
        score = 0
        for c in completion:
            c_str = self.tokenizer.decode(c, skip_special_tokens=True)
            score += substring_matching_judge_fn(c_str, detect_toks)
        return score / len(completion)
