import logging
import time
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from .models.base_metamodel import BaseMetaModel
from .tasks.proteingym.tasks import ProteinGymMetaSLTask

log = logging.getLogger("rich")

def run_metasupervised_evaluation(
    task: ProteinGymMetaSLTask,
    surrogate: BaseMetaModel,
    splits_to_evaluate: Tuple[str, ...] = ("validation",),
) -> Tuple[Dict[str, Union[float, int, np.number]], pd.DataFrame]:
    """
    Evaluate the surrogate model on specified splits of the task.

    Args:
        task (ProteinGymMetaSLTask): The task to evaluate.
        surrogate (BaseMetaModel): The surrogate model to be evaluated.
        splits_to_evaluate (Tuple[str, ...]): Dataset splits to evaluate on.
            Default is ("validation",).

    Returns:
        Tuple[Dict[str, Union[float, int, np.number]], pd.DataFrame]:
            - Metrics for the evaluated splits.
            - DataFrame containing sequences, target values, and predictions.
    """
    metrics = {}

    for split in splits_to_evaluate:
        t0 = time.time()

        # Evaluate surrogate on the current split
        split_metrics, preds, oracle_values, seqs = task.evaluate_surrogate(surrogate, split)
        split_metrics = dict(split_metrics)

        # Pad sequences to uniform length
        max_len = max(len(seq) for seq in seqs)
        padded_sequences = [seq.ljust(max_len, "-") for seq in seqs]

        # Prepare data dictionary for DataFrame
        data = {
            "sequence": padded_sequences,
            "target": oracle_values,
        }
        if preds:
            data["prediction"] = preds

        # Convert data to DataFrame
        predictions_df = pd.DataFrame(data)

        # Record evaluation time
        t1 = time.time()
        split_metrics["eval_time"] = t1 - t0

        # Update metrics with split-specific results
        for key, value in split_metrics.items():
            metrics[f"{split}_{key}"] = value

    return metrics, predictions_df
