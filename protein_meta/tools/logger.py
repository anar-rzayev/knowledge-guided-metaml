import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import neptune
import numpy as np
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig, listconfig
from s3fs import S3FileSystem

from protein_meta.logger import Logger

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")

class ListLogger(Logger):
    history: Dict[str, List[Union[np.ndarray, float, int]]] = {}

    def write(self, data: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        for key, value in data.items():
            if key in self.history:
                self.history[key].append(value)
            else:
                self.history[key] = [value]

    def close(self) -> None:
        pass

    def write_artifact(self, file_name: str) -> None:
        pass

    def get_checkpoint(self, file_path: str) -> Optional[str]:
        pass

class NoOpLogger(Logger):
    def write(self, data: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        pass

    def write_artifact(self, file_name: str) -> None:
        pass

    def close(self) -> None:
        pass

    def get_checkpoint(self, file_path: str) -> Optional[str]:
        pass

class NeptuneLogger(Logger):
    metadata = None

    def __init__(
        self,
        config: DictConfig,
        mode: str = "async",
        file_system: Optional[S3FileSystem] = None,
        neptune_tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.config = config
        self.file_system = file_system
        neptune_tags = neptune_tags or []

        if isinstance(self.config["logging"]["tags"], listconfig.ListConfig):
            tags = list(self.config["logging"]["tags"])
        else:
            assert (
                self.config["logging"]["tags"] is None
            ), "tags field must be None or list"
            tags = []
        tags += neptune_tags

        self.run = neptune.init_run(
            project=self.config["project_name"],
            tags=tags,
            mode=mode,
            **kwargs,
        )
        self.run["config"] = stringify_unsupported(self.config)
        self._t: int = 0

        if mode != "offline":
            self.run_id = self.run["sys/id"].fetch()
            NeptuneLogger.metadata = self.run_id

    def write(
        self,
        data: Mapping[str, float],
        label: str = "",
        timestep: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            self._t = timestep if timestep is not None else self._t + 1
            prefix = label and f"{label}/"
            for key, metric_value in data.items():
                if not np.isnan(metric_value):
                    self.run[f"{prefix}{key}"].log(
                        metric_value,
                        step=self._t,
                        wait=False,
                    )
        except Exception as e:
            log.error(f"Neptune Write Error: {e}")

    def write_artifact(self, file_name: str) -> None:
        try:
            logging.info("Saving checkpoint to Neptune")
            self.run[file_name].upload(file_name)
        except Exception as e:
            log.error(f"Neptune Write Artifact Error: {e}")

    def close(self) -> None:
        self.run.stop()

    def get_checkpoint(self, file_path: str) -> Optional[str]:
        try:
            parts = file_path.split("/")
            run = neptune.init_run(project=self.config["project"], with_id=parts[0])
            run[f"checkpoints/{parts[1]}"].download(
                destination=f"{parts[0]}_{parts[1]}"
            )
            return f"{parts[0]}_{parts[1]}"
        except Exception as e:
            log.info(f"Unable to load Checkpoint: {e}")
        return None

class TerminalLogger(Logger):
    def __init__(self, **kwargs: Any):
        print(">>> Terminal Logger")

    def write(
        self,
        data: Mapping[str, float],
        label: str = "",
        timestep: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        msg = "\n".join([f"{k}: {v:.5f}" for k, v in data.items()])
        if timestep is not None:
            print(f"{timestep:.2e}:\n{msg}\n")
        else:
            print(msg)

    def write_artifact(self, file_name: str) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def get_checkpoint(file_path: str) -> Optional[str]:
        pass

def logger_factory(
    logger_type: str,
    config_dict: DictConfig,
    mode: str = "async",
    file_system: Optional[S3FileSystem] = None,
    **kwargs: Any,
) -> Logger:
    if logger_type == "neptune":
        return NeptuneLogger(config_dict, mode=mode, file_system=file_system, **kwargs)
    elif logger_type == "terminal":
        return TerminalLogger(**kwargs)
    elif logger_type == "list":
        return ListLogger(**kwargs)
    else:
        raise ValueError(
            f"expected logger in ['neptune', 'terminal'], got {logger_type}."
        )

def get_logger_from_config(
    config: DictConfig, file_system: Optional[S3FileSystem], **kwargs: Any
) -> Logger:
    logger = logger_factory(
        config.logging.type,
        config,
        mode=config.logging.mode,
        file_system=file_system,
        **kwargs,
    )
    return logger
