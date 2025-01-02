from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from sklearn.model_selection import train_test_split

def create_all_single_mutants(
    candidate: Candidate,
    alphabet: str,
    mutation_range_start: Optional[int] = None,
    mutation_range_end: Optional[int] = None,
    include_null: bool = False,
) -> List[Candidate]:
    """Generate all single mutants for a given candidate sequence."""
    mutated_candidates = []
    mutation_range_start = mutation_range_start or 1
    mutation_range_end = mutation_range_end or len(candidate.sequence)
    for position, current_char in enumerate(
        candidate.sequence[mutation_range_start - 1 : mutation_range_end]
    ):
        for mutated_char in alphabet:
            if current_char != mutated_char:
                mutation = PointMutation(position, current_char, mutated_char)
                mutated_candidates.append(candidate.apply_mutation(mutation))
    if include_null:
        mutated_candidates.append(candidate)
    return mutated_candidates

@dataclass
class PointMutation:
    position: int
    from_char: str
    to_char: str

    def mutate(self, sequence: str) -> str:
        assert sequence[self.position] == self.from_char
        seq_list = list(sequence)
        seq_list[self.position] = self.to_char
        return "".join(seq_list)

    def __repr__(self) -> str:
        return f"{self.from_char}{self.position}{self.to_char}"

    @classmethod
    def from_str(cls, mutation_code: str) -> PointMutation:
        position = int(mutation_code[1:-1])
        from_char = mutation_code[0]
        to_char = mutation_code[-1]
        return cls(position, from_char, to_char)

@dataclass
class Candidate:
    sequence: str
    features: Any = None

    def __repr__(self) -> str:
        return f"Candidate(sequence={self.sequence})"

    def __len__(self) -> int:
        return len(self.sequence)

    def apply_mutation(self, mutation: PointMutation) -> Candidate:
        return Candidate(sequence=mutation.mutate(self.sequence))

    def apply_random_mutation(self, alphabet: str) -> Candidate:
        return self.apply_mutation(self.propose_random_mutation(alphabet))

    def propose_random_mutation(self, alphabet: str) -> PointMutation:
        position = np.random.choice(len(self.sequence))
        from_char = self.sequence[position]
        to_char = np.random.choice([c for c in alphabet if c != from_char])
        return PointMutation(position, from_char, to_char)

    def create_all_single_mutants(
        self,
        alphabet: str,
        mutation_range_start: Optional[int] = None,
        mutation_range_end: Optional[int] = None,
        include_null: bool = False,
    ) -> List[Candidate]:
        return create_all_single_mutants(
            self,
            alphabet,
            mutation_range_start=mutation_range_start,
            mutation_range_end=mutation_range_end,
            include_null=include_null,
        )
@dataclass
class SearchSpace:
    alphabet: str
    length: Optional[int] = None
    candidate_pool: Optional[List[Candidate]] = None

    @property
    def pool_sequences(self) -> Optional[List[str]]:
        if self.candidate_pool:
            return [cand.sequence for cand in self.candidate_pool]
        return None

    def check_is_valid(self, candidate: Candidate) -> None:
        if self.length:
            assert len(candidate) == self.length, f"Invalid length for candidate: {len(candidate)}"
        assert all(char in self.alphabet for char in candidate.sequence), "Invalid characters in candidate."
        if self.pool_sequences:
            assert candidate.sequence in self.pool_sequences, "Candidate not in pool."

    def update(self, scored_candidates: OraclePoints) -> None:
        for cand in scored_candidates.candidate_points:
            self.check_is_valid(cand)
        if self.candidate_pool:
            self.candidate_pool = [
                candidate for candidate in self.candidate_pool
                if candidate.sequence not in scored_candidates.sequences
            ]

class OraclePoints:
    def __init__(
        self, candidate_points: List[Candidate], oracle_values: np.ndarray
    ) -> None:
        self.candidate_points = candidate_points
        self._oracle_values = oracle_values
        assert len(candidate_points) == len(oracle_values), "Mismatch in candidates and oracle values length."

    @property
    def oracle_values(self) -> np.ndarray:
        return self._oracle_values

    @property
    def sequences(self) -> List[str]:
        return [cand.sequence for cand in self.candidate_points]

    def normalize(self, norm_str: Optional[str]) -> OraclePoints:
        assert self.oracle_values.size > 0, "Normalization requires non-empty oracle values."
        oracle_values = self.oracle_values
        if norm_str == "standardize":
            std = np.std(oracle_values) if len(self) > 1 else 1
            oracle_values = (oracle_values - oracle_values.mean()) / std
        elif norm_str == "normalize":
            oracle_values /= np.mean(np.abs(oracle_values))
        return OraclePoints(self.candidate_points, oracle_values)

    def __len__(self) -> int:
        return len(self.candidate_points)

@dataclass
class OptimizerState:
    search_space: SearchSpace
    datasets: Dict[str, OraclePoints]
    val_frac: float

    @property
    def train_dataset(self) -> OraclePoints:
        return self.datasets["train"]

    @property
    def validation_dataset(self) -> OraclePoints:
        return self.datasets["validation"]

    def add_to_training_dataset(self, scored_candidates: OraclePoints) -> None:
        if self.val_frac > 0:
            train_candidates, val_candidates, train_values, val_values = train_test_split(
                scored_candidates.candidate_points,
                scored_candidates.oracle_values,
                test_size=self.val_frac,
            )
            self.train_dataset.append(train_candidates, train_values)
            self.validation_dataset.append(val_candidates, val_values)
        else:
            self.train_dataset.append(scored_candidates.candidate_points, scored_candidates.oracle_values)
        self.search_space.update(scored_candidates)

    def summary(self) -> Dict[str, Union[int, float]]:
        summary = {
            "num_train": len(self.train_dataset),
            "num_validation": len(self.validation_dataset),
        }
        if self.search_space.candidate_pool:
            summary["candidate_pool_size"] = len(self.search_space.candidate_pool)
        return summary