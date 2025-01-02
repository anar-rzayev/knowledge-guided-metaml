import logging
import os
import random
import re
from typing import Generator, List, Optional, TextIO, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from protein_meta.constants import BASEDIR

log = logging.getLogger("rich")

def select_computation_device(gpu_index: Optional[int] = None, use_cpu_only: bool = False) -> torch.device:
    """Determine the device (CPU or GPU) for computation."""
    if torch.cuda.is_available() and not use_cpu_only:
        return torch.device(f"cuda:{gpu_index}" if gpu_index is not None else "cuda")
    return torch.device("cpu")

def log_environment_details(config: DictConfig) -> None:
    """Log information about the current environment and configuration."""
    log.info(f"Input Path: {os.environ.get('AICHOR_INPUT_PATH')}")
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Configuration details:\n{OmegaConf.to_yaml(config)}")

def _parse_fasta_lines(
    lines: TextIO,
    retain_gaps: bool = True,
    retain_insertions: bool = True,
    convert_to_uppercase: bool = False,
) -> Generator[Tuple[str, str], None, None]:
    """Extract sequence details from FASTA format lines."""
    sequence, description = None, None

    def clean_sequence(seq: str) -> str:
        if not retain_gaps:
            seq = re.sub("-", "", seq)
        if not retain_insertions:
            seq = re.sub(r"[a-z.]", "", seq)
        return seq.upper() if convert_to_uppercase else seq

    for line in lines:
        if line.startswith(">"):
            if sequence is not None:
                yield description, clean_sequence(sequence)
            description = line.strip()[1:]
            sequence = ""
        else:
            sequence = (sequence or "") + line.strip()

    if sequence and description:
        yield description, clean_sequence(sequence)

def fasta_reader(
    filepath: str,
    encoding: Optional[str] = None,
    retain_insertions: bool = True,
    retain_gaps: bool = True,
    convert_to_uppercase: bool = False,
) -> Generator[Tuple[str, str], None, None]:
    """Generate sequences and descriptions from a FASTA file."""
    with open(filepath, "r", encoding=encoding) as file:
        yield from _parse_fasta_lines(
            file, retain_gaps=retain_gaps, retain_insertions=retain_insertions, convert_to_uppercase=convert_to_uppercase
        )

def load_fasta(
    filepath: str,
    encoding: Optional[str] = None,
    retain_insertions: bool = True,
    retain_gaps: bool = True,
    convert_to_uppercase: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load FASTA file and return lists of sequence names and sequences."""
    names, sequences = [], []
    for name, seq in fasta_reader(
        filepath, encoding, retain_insertions, retain_gaps, convert_to_uppercase
    ):
        names.append(name)
        sequences.append(seq)
    return names, sequences

def load_msa(msa_filepath: str, msa_format: str) -> Tuple[List[str], List[str]]:
    """Load sequences from a multiple sequence alignment (MSA) file."""
    if msa_format == "a3m":
        return load_fasta(msa_filepath, retain_insertions=False, convert_to_uppercase=True)
    if msa_format == "gym":
        return load_fasta(msa_filepath, retain_insertions=True, convert_to_uppercase=True)
    raise ValueError(f"Unsupported MSA format: {msa_format}")

def fetch_git_commit_hash() -> str:
    """Retrieve the current Git commit hash."""
    git_sha = os.getenv("VCS_SHA")
    if git_sha:
        return git_sha
    try:
        with open(os.path.join(BASEDIR, ".git/HEAD")) as head_file:
            ref_path = head_file.read().strip().split()[1]
        with open(os.path.join(BASEDIR, f".git/{ref_path}")) as ref_file:
            return ref_file.read().strip()
    except (FileNotFoundError, IndexError) as err:
        log.warning(f"Git commit hash could not be determined: {err}")
        return "unknown"

def initialize_random_seeds(seed_value: int = 42) -> None:
    """Set seeds for reproducibility across libraries."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    log.info(f"Random seed initialized to {seed_value}")
