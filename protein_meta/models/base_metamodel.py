import abc
import logging
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
from numpy import number

from protein_meta.dataclasses import Candidate, OraclePoints
from protein_meta.logger import Logger

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class BaseMetaModel(abc.ABC):
    def __init__(
        self,
        name: str,
        support_size: Optional[int],  # Number of support set datapoints to pass as context (shot). None for any size.
        query_size: Optional[int],  # Number of query set datapoints to pass as targets. None for any size.
        use_all_data: bool,  # If True, all data is in support or query set. Ignored if sizes given.
        max_context_sz: int,  # Maximum size of dataset for training. (Will subsample if larger.)
        num_outputs: int = 1,
    ) -> None:
        """Base class for meta-learning models.
        
        Args:
            name: Name of the model
            support_size: Number of samples in support set (for few-shot learning)
            query_size: Number of samples in query set (for evaluation)
            use_all_data: Whether to use all available data for training
            max_context_sz: Maximum context size for limiting memory usage
            num_outputs: Number of output dimensions predicted for each sample
        """
        self.name = name
        self.num_outputs = num_outputs
        self.support_size = support_size
        self.query_size = query_size
        self.max_context_sz = max_context_sz
        self.use_all_data = use_all_data
        self.metadata: Dict[str, Dict] = {}
        self.save_dir: Optional[str] = None

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Delete any temporary files, checkpoints etc that aren't being persisted."""
        pass

    def set_dir(self, save_dir: str) -> None:
        """Set directory for saving model checkpoints and artifacts."""
        self.save_dir = save_dir

    @abc.abstractmethod
    def fit(
        self,
        train_data: Dict[str, OraclePoints],
        seed: int,
        logger: Logger,
        eval_func: Callable,
        num_steps: Optional[int] = None,
    ) -> None:
        """Train meta-learning model on multiple datasets.
        
        Args:
            train_data: Dictionary mapping task names to training data
            seed: Random seed for reproducibility
            logger: Logger for tracking metrics
            eval_func: Function for evaluation during training
            num_steps: Number of training steps (optional)
        """
        pass

    def generate_support_chunked_query_splits(
        self,
        task_data: Dict[str, OraclePoints],
        support_size: int,
        query_size: int,
        early_stop_size: int,
        eval_size: int,
        num_evals: int,
        random_st: np.random.RandomState,
        normalization: Optional[str],
        allow_partial_query_set: bool,
    ) -> Generator[Tuple[OraclePoints, OraclePoints, str, OraclePoints], None, None]:
        """Generate fixed splits for validation with chunked queries.
        
        Args:
            task_data: Dictionary mapping task names to data
            support_size: Size of support set
            query_size: Size of query set
            early_stop_size: Size of early stopping set
            eval_size: Size of evaluation set
            num_evals: Number of evaluation runs
            random_st: Random state for reproducibility
            normalization: Type of normalization to apply
            allow_partial_query_set: Whether to allow partial query sets
        """
        for _ in range(num_evals):
            for task_name, oracle_points in task_data.items():
                oracle_points = oracle_points.normalize(normalization)
                effective_eval_size = eval_size
                expected_size = support_size + early_stop_size + eval_size
                min_size = support_size + early_stop_size
                if not allow_partial_query_set:
                    min_size += query_size
                if len(oracle_points) < min_size:
                    raise ValueError(
                        f"Expected at least {min_size} data points,"
                        f" but got {len(oracle_points)}"
                    )
                if len(oracle_points) < expected_size:
                    print(
                        f"Warning: Expected at least {expected_size} data points,"
                        f" but got {len(oracle_points)}"
                    )
                    effective_eval_size = (
                        len(oracle_points) - support_size - early_stop_size
                    )
                    if not allow_partial_query_set:
                        effective_eval_size = (
                            effective_eval_size // query_size
                        ) * query_size
                        if effective_eval_size == 0:
                            raise ValueError(
                                "Expected enough data for at least one query set."
                            )
                        assert (
                            effective_eval_size % query_size == 0
                        ), "eval_size must be divisible by query_size"
                permuted_data = oracle_points.permutation(random_st)
                support_set = permuted_data[:support_size]
                early_stop_set = permuted_data[
                    support_size : support_size + early_stop_size
                ]
                eval_set = permuted_data[
                    support_size
                    + early_stop_size : support_size
                    + early_stop_size
                    + effective_eval_size
                ]
                yield support_set, eval_set, task_name, early_stop_set

    def generate_support_query_splits(
        self,
        task_data: Dict[str, OraclePoints],
        random_st: np.random.RandomState,
        num_evals: Optional[int],
        normalization: Optional[str],
    ) -> Generator[Tuple[OraclePoints, OraclePoints, str, None], None, None]:
        """Generate splits for training and evaluation.
        
        Args:
            task_data: Dictionary mapping task names to data
            random_st: Random state for reproducibility
            num_evals: Number of evaluation runs (optional) 
            normalization: Type of normalization to apply
        """
        tasks = list(task_data.items())
        eval_num = 0
        while num_evals is None or eval_num < num_evals:
            perm = random_st.permutation(len(tasks))
            for i in perm:
                task_name, dataset = tasks[i]
                dataset = dataset.normalize(normalization)

                cur_support_size = self.support_size
                cur_query_size = self.query_size
                permuted_data = dataset.permutation(random_st)
                if len(permuted_data) > self.max_context_sz:
                    permuted_data = permuted_data[: self.max_context_sz]
                if self.support_size is None and self.query_size is None:
                    assert (
                        self.use_all_data
                    ), "Must specify support_size and/or query_size if use_all_data is False."
                    split = random_st.randint(
                        0,
                        len(permuted_data) - 1,
                    )
                    support_set, query_set = (
                        permuted_data[:split],
                        permuted_data[split:],
                    )
                elif self.support_size is None:
                    max_support_size = len(permuted_data) - self.query_size
                    cur_support_size = (
                        max_support_size
                        if self.use_all_data
                        else random_st.randint(
                            0, max_support_size
                        )
                    )
                elif self.query_size is None:
                    max_query_size = len(permuted_data) - self.support_size
                    cur_query_size = (
                        max_query_size
                        if self.use_all_data
                        else random_st.randint(1, max_query_size)
                    )
                cur_support_size = cast(int, cur_support_size)
                cur_query_size = cast(int, cur_query_size)
                support_set, query_set = (
                    permuted_data[:cur_support_size],
                    permuted_data[cur_support_size : cur_support_size + cur_query_size],
                )

                yield support_set, query_set, task_name, None

            eval_num += 1

    @abc.abstractmethod
    def _predict(
        self,
        support_set: OraclePoints,
        query_set: List[Candidate],
        task_name: str,
        early_stop_set: Optional[OraclePoints] = None,
        return_params: bool = False,
    ) -> Union[np.ndarray, List[torch.nn.Parameter]]:
        """Make predictions for query points given support set.
        
        Args:
            support_set: Support set data
            query_set: Query points to predict
            task_name: Name of current task
            early_stop_set: Data for early stopping (optional)
            return_params: Whether to return model parameters
            
        Returns:
            Predictions for query points or model parameters
        """
        pass

    def check_predictions_shape(
        self, candidate_points: List[Candidate], predictions: np.ndarray
    ) -> None:
        """Verify that predictions have expected shape."""
        expected_shape = (
            (len(candidate_points),)
            if self.num_outputs == 1
            else (len(candidate_points), self.num_outputs)
        )
        assert predictions.shape == expected_shape, (
            f"Expected _predict to return array with shape ({expected_shape}), "
            f"got shape {predictions.shape}"
        )

    def set_metadata(self, task_metadata: Dict[str, Dict]) -> None:
        """Set metadata for tasks."""
        self.metadata = task_metadata

    def predict(
        self,
        support_set: OraclePoints,
        query_set: List[Candidate],
        task_name: str,
        early_stop_set: Optional[OraclePoints] = None,
    ) -> np.ndarray:
        """Make predictions for query points."""
        predictions = self._predict(
            support_set, query_set, task_name, early_stop_set=early_stop_set
        )
        self.check_predictions_shape(query_set, predictions)
        return predictions

    @abc.abstractmethod
    def get_training_summary_metrics(
        self,
    ) -> Mapping[str, Union[float, int, number]]:
        """Get summary metrics from training."""
        pass