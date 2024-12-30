# Knowledge-Guided Meta-Learning for Protein Fitness Prediction

This repository implements a meta-learning framework for predicting protein fitness. By leveraging protein language models (PLMs) and integrating domain knowledge, the framework excels in low-data scenarios and achieves competitive results on benchmark datasets like ProteinGym.

---

## 🚀 Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anar-rzayev/knowledge-guided-metaml.git
   cd knowledge-guided-metaml
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

1. **Train the Meta-Learner:**
   ```bash
   python train_meta.py --config configs/meta_config.yaml
   ```

2. **Fine-Tune for a New Task:**
   ```bash
   python fine_tune.py --task new_task.yaml
   ```

3. **Evaluate the Model:**
   ```bash
   python evaluate.py --dataset protein_gym --config configs/eval_config.yaml
   ```

## 📁 Project Structure

```
├── configs/        # Configuration files
├── data/          # Data processing utilities
├── models/        # Model architectures
├── utils/         # Helper functions
└── scripts/       # Training and evaluation scripts
```

## 🛠️ Implementation Details

The framework implements several key components:

- **ESM2 Embeddings**: Utilizes ESM2-8M for protein sequence embeddings
- **Knowledge Integration**: Axial attention mechanism for incorporating domain knowledge
- **Sub-sampled Fine-tuning**: Efficient adaptation to new tasks while preventing memorization
- **Preference-based Learning**: Ranking loss for better fitness prediction

## 🔧 Configuration

Key hyperparameters can be modified in `configs/meta_config.yaml`:

```yaml
model:
  hidden_dim: 768
  n_layers: 5
  n_heads: 4
  
training:
  batch_size: 4
  lr: 6e-5
  max_steps: 50000
```
