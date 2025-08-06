from __future__ import annotations

import logging
from random import Random
from typing import Any, Optional


eval_logger = logging.getLogger(__name__)


class ContextSampler:
    def __init__(
        self,
        docs: list[dict[str, Any]],
        *,
        rnd: int = 1234,
        fewshot_indices: list[int] | None = None,
    ) -> None:
        self.rnd = Random(rnd)
        self.docs = docs
        self.fewshot_indices = fewshot_indices

        if self.fewshot_indices and self.docs:
            self.docs = [self.docs[i] for i in self.fewshot_indices]

    def sample(
        self, n: int, exclude: Optional[dict[str, Any]] = None, **kwargs
    ) -> list[dict]:
        """
        Sample n documents from the pool.

        Args:
            n: Number of documents to sample
            exclude: Optional document to exclude from sampling

        Returns:
            List of sampled documents
        """
        # Filter out excluded document if specified
        sample_size = min(n, len(self.docs))
        if sample_size == 0:
            return []
        return (
            self.rnd.sample(self.docs, sample_size)
            if not exclude
            else [
                doc
                for doc in self.rnd.sample(self.docs, sample_size + 1)
                if doc != exclude
            ]
        )

    def set_rnd(self, rnd: int) -> None:
        self.rnd = Random(rnd)


class FirstNSampler(ContextSampler):
    def sample(self, n: int, **kwargs) -> list[dict]:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), (
            f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        )
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n: int, **kwargs) -> list[dict]:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n: int, **kwargs) -> None:
        """ """
        pass


SAMPLER_REGISTRY: dict[str, type[ContextSampler]] = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name: str):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )
