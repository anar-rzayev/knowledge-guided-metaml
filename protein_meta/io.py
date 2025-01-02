import json
import os
import pickle
from typing import Any, Dict, List, Optional, TextIO

import numpy as np
import pandas as pd
import torch
import yaml
from s3fs.core import S3FileSystem

def load_yaml(yaml_file: str) -> Any:
    """Load data from a YAML file."""
    with open(yaml_file, "r") as yf:
        return yaml.load(yf, Loader=yaml.UnsafeLoader)

def load_json(jsonfile: str) -> Any:
    """Load data from a JSON file."""
    with open(jsonfile, "r") as jf:
        return json.load(jf)

class FileHandler:
    """Handles file operations locally or with S3, depending on the provided endpoint."""

    def __init__(self, s3_endpoint: Optional[str] = None, bucket: Optional[str] = "input") -> None:
        self.s3_endpoint = s3_endpoint
        self.bucket = bucket
        if s3_endpoint:
            self.s3 = S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint})
            self.bucket_path = os.environ.get(f"AICHOR_{bucket.upper()}_PATH", "./")
        else:
            self.bucket_path = "./"

    def expand_path(self, path: str) -> str:
        """Generate full path based on bucket settings."""
        return os.path.join(self.bucket_path, path)

    def read_numpy(self, path: str) -> np.ndarray:
        """Read a NumPy array from a file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path)) as f:
                return np.load(f, allow_pickle=True)
        return np.load(path, allow_pickle=True)

    def save_numpy(self, path: str, array: np.ndarray) -> None:
        """Save a NumPy array to a file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path), "wb") as f:
                f.write(pickle.dumps(array))
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, array)

    def read_text(self, path: str) -> List[str]:
        """Read lines from a text file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path)) as f:
                return f.readlines()
        with open(self.expand_path(path)) as f:
            return f.readlines()

    def save_text(self, path: str, lines: List[str]) -> None:
        """Save lines to a text file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path), "w") as f:
                f.writelines(lines)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(self.expand_path(path), "w") as f:
                f.writelines(lines)

    def read_json(self, path: str) -> Dict:
        """Read JSON data from a file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path)) as f:
                return json.load(f)
        with open(self.expand_path(path)) as f:
            return json.load(f)

    def save_json(self, path: str, data: Dict) -> None:
        """Save JSON data to a file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path), "w") as f:
                json.dump(data, f, indent=4)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(self.expand_path(path), "w") as f:
                json.dump(data, f, indent=4)

    def read_yaml(self, path: str) -> Dict:
        """Read YAML data from a file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path)) as f:
                return yaml.load(f, Loader=yaml.UnsafeLoader)
        with open(self.expand_path(path)) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)

    def save_yaml(self, path: str, data: Dict) -> None:
        """Save YAML data to a file."""
        if self.s3_endpoint:
            with self.s3.open(self.expand_path(path), "w") as f:
                yaml.dump(data, f)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(self.expand_path(path), "w") as f:
                yaml.dump(data, f)

    def read_csv(self, path: str, header: str = "infer") -> pd.DataFrame:
        """Read CSV data into a pandas DataFrame."""
        return pd.read_csv(self.expand_path(path), header=header)

    def save_csv(self, path: str, df: pd.DataFrame, header: bool = True, index: bool = False) -> None:
        """Save a pandas DataFrame to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(self.expand_path(path), index=index, header=header)

    def listdir(self, path: str) -> List[str]:
        """List files in a directory."""
        if self.s3_endpoint:
            return [file.split(os.path.sep)[-1] for file in self.s3.ls(self.expand_path(path))]
        return os.listdir(path)

    def isfile(self, path: str) -> bool:
        """Check if a path points to a file."""
        if self.s3_endpoint:
            return self.s3.isfile(self.expand_path(path))
        return os.path.isfile(path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create directories."""
        if not self.s3_endpoint:
            os.makedirs(path, exist_ok=exist_ok)

    def download(self, remote_path: str, local_path: str, recursive: bool = False, **kwargs: Any) -> None:
        """Download files from S3 to local."""
        assert self.s3_endpoint
        self.s3.download(self.expand_path(remote_path), local_path, recursive=recursive, **kwargs)

    def upload(self, local_path: str, remote_path: str, recursive: bool = False, **kwargs: Any) -> None:
        """Upload files from local to S3."""
        assert self.s3_endpoint
        self.s3.put(local_path, self.expand_path(remote_path), recursive=recursive, **kwargs)

input_handler = FileHandler(os.environ.get("S3_ENDPOINT"), bucket="input")
