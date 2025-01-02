import gc
import logging
import os
import time
from typing import Callable, Dict, Any

import hydra
import torch
from omegaconf import DictConfig, open_dict
from neptune.integrations.python_logger import NeptuneHandler

# Configure basic logging
logging.basicConfig(
    level="NOTSET",
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("experiment")

class ExperimentRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_environment()
        
    def setup_environment(self) -> None:
        """Initialize experiment environment"""
        from protein_meta.utils import display_info, set_seeds, get_current_git_commit_hash
        from protein_meta.tools.logger import get_logger_from_config
        
        display_info(self.cfg)
        set_seeds(self.cfg.seed)
        self.commit = get_current_git_commit_hash()
        logger.info(f"Git commit: {self.commit}")
        
        with open_dict(self.cfg):
            self.cfg.commit_hash = self.commit
            
    def initialize_task(self):
        """Setup task and datasets"""
        self.task = hydra.utils.instantiate(self.cfg.task)
        logger.info("Initializing task and datasets...")
        self.task.setup_datasets(
            load_zero_shot=self.cfg.surrogate.model_config.aux_pred
        )
        
    def setup_logging(self):
        """Configure experiment logging"""
        from protein_meta.tools.logger import get_logger_from_config
        
        neptune_tags = [
            str(self.cfg.task.task_name),
            str(self.cfg.surrogate.name),
            str(type(self.task)),
            self.commit
        ]
        
        self.exp_logger = get_logger_from_config(
            self.cfg, 
            file_system=None,
            neptune_tags=neptune_tags
        )
        
        if self.cfg.logging.type == "neptune":
            logger.addHandler(NeptuneHandler(run=self.exp_logger.run))

    def initialize_surrogate(self):
        """Setup surrogate model"""
        logger.info("Setting up surrogate model...")
        self.surrogate = hydra.utils.instantiate(self.cfg.surrogate, _recursive_=True)
        
        run_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "results",
            "".join(self.cfg.logging.tags),
            str(self.cfg.seed)
        )
        self.surrogate.set_dir(run_path)
        
        if self.task.has_metadata:
            logger.info("Loading task metadata...")
            self.surrogate.set_metadata(self.task.metadata)

    def evaluate_model(self, label: str = "metrics") -> Dict[str, float]:
        """Run model evaluation"""
        from protein_meta.evaluation import run_metasupervised_evaluation
        
        metrics, _ = run_metasupervised_evaluation(self.task, self.surrogate)
        logger.info(
            "Evaluation results:\n\t" + 
            "\n\t".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
        return metrics

    def train_model(self):
        """Execute model training"""
        logger.info("Starting training phase...")
        logger.info("Dataset sizes: " + " ".join(
            f"{split}: {len(self.task.data_splits[split])}"
            for split in ["train", "validation"]
            if split in self.task.data_splits
        ))
        
        start_time = time.time()
        eval_fn = lambda s: run_metasupervised_evaluation(self.task, s)
        
        self.surrogate.fit(
            self.task.data_splits["train"],
            self.cfg.seed,
            logger=self.exp_logger,
            eval_func=eval_fn
        )
        
        if self.cfg.evaluate_end:
            train_time = time.time() - start_time
            final_metrics = self.evaluate_model()
            final_metrics.update(self.surrogate.get_training_summary_metrics())
            final_metrics.update(self.task.data_summary())
            final_metrics["train_time"] = train_time
            self.exp_logger.write(final_metrics, label="end_metrics")

    def cleanup(self):
        """Cleanup resources"""
        self.exp_logger.close()
        self.surrogate.cleanup()
        del self.surrogate
        del self.task
        torch.cuda.empty_cache()
        gc.collect()

@hydra.main(
    config_path="config",
    config_name="metasupervised",
    version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    runner = ExperimentRunner(cfg)
    runner.initialize_task()
    runner.setup_logging()
    runner.initialize_surrogate()
    
    if cfg.evaluate_first:
        logger.info("Running initial evaluation...")
        init_metrics = runner.evaluate_model()
        runner.exp_logger.write(init_metrics, label="init_metrics", timestep=0)
        
        if cfg.exit_after_first_eval:
            runner.cleanup()
            logger.info("Exiting after initial evaluation")
            return
    
    runner.train_model()
    runner.cleanup()
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
