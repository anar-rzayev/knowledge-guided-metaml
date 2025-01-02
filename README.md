# 🧬 Knowledge-Guided Meta-Learning for Protein Fitness Prediction

This repository explores knowledge-guided meta-learning for protein fitness prediction. The approach investigates combining meta-learning and domain knowledge for improved protein fitness prediction.

## ⚡ Quick Start

To setup and execute a meta-learning experiment, use:

```bash

bash ./run_meta_supervised.sh

```

The config files can be found at:

- config/meta_supervised.yaml (experiment settings)

- config/task/gym_supervised.yaml (evaluation)

- config/model/meta.yaml (model and training)

For debugging purposes:

```bash

python run_metasupervised.py experiment_group=test logging.type=terminal surrogate.train_config.batch_sz=2 surrogate.train_config.support_size=2 surrogate.train_config.query_size=4

```

## 📂 Project Structure

```

├── config/              # Configuration files

├── docker/             # Dockerfile and environment setup

├── protein_meta/       # Core implementation

├── requirements.txt    # Dependencies

├── run_meta_supervised.sh    # Training script

└── run_metasupervised.py    # Main execution script

```

## 🛠️ Requirements

The code was tested with Python 3.10.11. Install requirements:

```bash

pip install -r requirements.txt

```

## ✨ Acknowledgments

This project was developed to explore the integration of domain knowledge with meta-learning approaches for protein fitness prediction.

📝 Note: This is a proof-of-concept implementation for research exploration purposes.
