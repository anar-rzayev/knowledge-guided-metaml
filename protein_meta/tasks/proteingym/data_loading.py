import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

import datasets
import numpy as np
import pandas as pd

from protein_meta.constants import (
    SUBS_ZERO_SHOT_COLS,
    SUBS_ZERO_SHOT_COLS_to_index,
)
from protein_meta.tasks.utils import FitnessTaskMetadata

# Wild-type values for specific datasets
WT_VALUES = {
    "GFP_AEQVI_Sarkisyan_2016": 3.72,
    "CAPSD_AAV2S_Sinai_substitutions_2021": -0.918194,
}

# Logging setup
logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")

# Load substitution metadata
subs_meta = pd.read_csv(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "DMS_substitutions.csv")
).set_index("DMS_id")

def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize a NumPy array."""
    return (x - x.mean()) / x.std()

def get_gym_metadata(dms_name: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific dataset."""
    return subs_meta.loc[dms_name].to_dict()

def make_msa_filepath(msa_file: str) -> str:
    """Construct the file path for an MSA file."""
    return f"ProteinGym/MSA_files/{msa_file}"

def make_pdb_filepath(pdb_file: str) -> str:
    raise NotImplementedError()

def make_gym_metadata(dms_name: str) -> FitnessTaskMetadata:
    """Generate metadata for a DMS dataset."""
    metadata = get_gym_metadata(dms_name)
    msa_file = make_msa_filepath(metadata["MSA_filename"])
    assert os.path.isfile(msa_file), f"MSA file {msa_file} not found"
    return FitnessTaskMetadata(
        dms_name=dms_name,
        wt_sequence=metadata["target_seq"],
        msa_file=msa_file,
        msa_format="gym",
    )

def get_from_harvard(dms_name: str, download_zero_shot: bool = False) -> str:
    """Download and manage ProteinGym data from the Harvard repository."""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    gym_dir = os.path.join(base_dir, "data/ProteinGym/")
    harvard_dir = os.path.join(gym_dir, "ProteinGym_v1.1")
    harvard_file = os.path.join(gym_dir, "ProteinGym_v1.1.zip")
    
    os.makedirs(gym_dir, exist_ok=True)

    if not os.path.exists(harvard_file):
        log.info("Downloading ProteinGym data from Harvard repository.")
        url = "https://zenodo.org/records/13936340/files/ProteinGym_v1.1.zip?download=1"
        os.system(f"wget {url} -O {harvard_file}")
    
    if not os.path.exists(harvard_dir):
        os.system(f"unzip {harvard_file} -d {gym_dir}")
    
    subs_dir = os.path.join(harvard_dir, "DMS_ProteinGym_substitutions")
    if not os.path.exists(f"{subs_dir}/{dms_name}.csv"):
        os.system(f"unzip {harvard_file} -d {gym_dir}")
        assert os.path.exists(
            f"{subs_dir}/{dms_name}.csv"
        ), f"Dataset {dms_name} not found in {subs_dir} after unzipping."
    
    return f"{subs_dir}/{dms_name}.csv"

def load_gym_dataset(
    dms_name: str, zero_shot_dms_names: Set[str] = set()
) -> datasets.Dataset:
    """Load a ProteinGym dataset as a Hugging Face Dataset.

    Args:
        dms_name (str): Name of the DMS dataset.
        zero_shot_dms_names (Set[str]): Names of datasets with zero-shot predictions.

    Returns:
        datasets.Dataset: Loaded dataset.
    """
    dms_data_path = get_from_harvard(dms_name)
    hf_data = datasets.load_dataset("csv", data_files=dms_data_path)["train"]

    if dms_name not in zero_shot_dms_names:
        return hf_data

    zero_shot_path = get_from_harvard(dms_name, download_zero_shot=True)
    zero_shot_df = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    zero_shot_path, model.split("_L")[0], model, f"{dms_name}.csv"
                )
                if "_L" in model
                else os.path.join(zero_shot_path, model, f"{dms_name}.csv")
            )[SUBS_ZERO_SHOT_COLS_to_index[model]]
            for model in SUBS_ZERO_SHOT_COLS
        ],
        axis=1,
    )

    standardized_cols = [f"standardized_{col}" for col in SUBS_ZERO_SHOT_COLS]
    zero_shot_df[standardized_cols] = standardize(zero_shot_df[SUBS_ZERO_SHOT_COLS])

    hf_data_df = hf_data.to_pandas()
    zero_shot_df = zero_shot_df.iloc[: len(hf_data_df)]
    combined_df = pd.concat([hf_data_df, zero_shot_df], axis=1)

    return datasets.Dataset.from_pandas(combined_df)


def count_num_mutations(mutant_code: str) -> int:
    return len(mutant_code.split(":"))

def subsample_dataset(
    input_dataset: datasets.Dataset,
    n_rows: int,
    seed: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
) -> datasets.Dataset:
    """
    Sample a specified number of rows from a Hugging Face dataset.

    Args:
        input_dataset (datasets.Dataset): The original dataset to sample from.
        n_rows (int): The number of rows to sample.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        datasets.Dataset: A new dataset containing the sampled rows.
    """
    # Shuffle the input dataset
    shuffled_dataset = input_dataset.shuffle(seed=seed, generator=generator)

    # Select the first n_rows from the shuffled dataset
    sampled_dataset = shuffled_dataset.select(list(range(n_rows)))

    return sampled_dataset

def subsample_splits(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    num_train: int,
    num_test: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    # TODO handle num_train / num_test greater than dataset size
    assert num_train is not None
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None
    train_dataset = subsample_dataset(train_dataset, num_train, generator=rng)
    if num_test is not None:
        test_dataset = subsample_dataset(test_dataset, num_test, generator=rng)
    return train_dataset, test_dataset

def split_dataset(
    dataset: datasets.Dataset,
    num_train: int,  # TODO: allow None for non-random splits?
    num_test: Optional[int] = None,
    split_type: str = "random",
    dms_name: Optional[str] = None,
    seed: Optional[int] = None,
    minimum_test_set_size: int = 32,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Split dataset based on configured splits.

    Split types are based on split types used in the paper:
    `Benchmark tasks in fitness landscape inference for proteins` (FLIP)

    Currently supported:
        random: random train / test split
        low_vs_high: train on sequences with DMS_score < WT, test on higher
        one_vs_many: train on single mutants, test on multi-mutants

    Args:
        dataset: datasets.Dataset
        num_train: int number of training points
        num_test: number of test points. If None, set automatically to
            complement of num_train
        split_type: type of split (random, low_vs_high, one_vs_many)
        dms_name: name of dataset in ProteinGym. Only required for low_vs_high
            split_type. dms_name is value in DMS_id field of ProteinGym
            reference files.
        seed: random seed to use for splitting.
        minimum_test_set_size: if the training set size and the test set size are
            greater than the dataset size, we automatically adjust the test
            set size, raising an exception if there are fewer than this
            number of sequences in the test set (i.e. the complement of the
            training set)
    """
    if split_type == "random":
        if num_test is None:
            assert num_train is not None  # would lead to unexpected behaviour
        if num_test is not None and num_train + num_test > len(dataset):
            assert num_train < (
                len(dataset) - minimum_test_set_size
            ), f"num_train too large relative to dataset size ({num_train}) vs ({len(dataset)})"
            num_test = len(dataset) - num_train
            print(
                "Warning: train and test sizes combined are greater than dataset size: "
                f"automatically re-setting num_test to len(dataset)-num_train={num_test}"
            )
        splits = dataset.train_test_split(
            test_size=num_test, train_size=num_train, seed=seed
        )
        return splits["train"], splits["test"]
    elif split_type == "one_vs_many":
        condition = dataset["mutant"].map(count_num_mutations) == 1
        train_dataset = dataset.filter(condition)
        test_dataset = dataset.filter(condition)
        # TODO write test
        return subsample_splits(
            train_dataset,
            test_dataset,
            num_train=num_train,
            num_test=num_test,
            seed=seed,
        )
    else:
        assert (
            dms_name is not None and dms_name in WT_VALUES
        ), f"no wt value for dms_name {dms_name}"
        condition = dataset["DMS_score"] < WT_VALUES[dms_name]
        train_dataset = dataset.filter(condition)
        test_dataset = dataset.filter(~condition)
        return subsample_splits(
            train_dataset,
            test_dataset,
            num_train=num_train,
            num_test=num_test,
            seed=seed,
        )
