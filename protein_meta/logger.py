import abc
from typing import Any, Mapping, Optional

class Logger(abc.ABC):
    """Abstract base class for a logger with various functionalities."""

    @abc.abstractmethod
    def write(self, data: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        """Writes `data` to the desired destination (file, terminal, etc.)."""
        pass

    @abc.abstractmethod
    def write_artifact(self, file_name: str) -> None:
        """Saves an artifact (e.g., file) to the logging destination."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger and ensures no further writes are expected."""
        pass

    @abc.abstractmethod
    def get_checkpoint(self, file_path: str) -> Optional[str]:
        """Fetches a checkpoint file by its path or identifier."""
        pass

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Optional[Exception], exc_val: Optional[Exception], exc_tb: Optional[Exception]) -> None:
        self.close()
