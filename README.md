# NBA Shot Prediction using Hybrid Attention-LSTM and XGBoost-

A deep learning and machine learning based sports analytics project focused on predicting NBA shot success using temporal sequence learning, attention mechanisms, and hybrid ensemble modeling.

#  Project Overview-

Predicting whether a basketball shot will be successful is a challenging sports analytics problem because shot outcomes depend on multiple contextual and temporal factors such as:

- Shot distance
- Defender pressure
- Player momentum
- Recent shooting history
- Game situation
- Shot clock
- Touch time

Traditional machine learning models handle structured tabular data effectively but fail to capture sequential player behavior across consecutive shots.

This project proposes a hybrid framework combining:

- Attention-based LSTM (temporal learning)
- XGBoost (tabular feature learning)
- Hybrid ensemble strategies

The system learns both:
- Temporal shooting patterns
- Contextual basketball statistics

to improve NBA shot success prediction.

# Key Features-

- Complete NBA preprocessing pipeline
- Temporal sequence generation
- Rolling momentum feature engineering
- Attention-based LSTM architecture
- XGBoost implementation
- Hybrid weighted ensemble
- Hybrid stacking ensemble
- Multiple visualization outputs
- Research-paper-ready experimentation
- IEEE-style research implementation


# Dataset Information-

The project uses publicly available NBA datasets.

## Datasets Used

| Dataset | Description |
| `shot_logs.csv` | Shot-level NBA shot records |
| `games.csv` | Game metadata |
| `games_details.csv` | Player statistics |
| `ranking.csv` | Team rankings |
| `players.csv` | Player information |
| `teams.csv` | Team information |

# Final Dataset Statistics-

| Metric | Value |
| Total shots | 128,069 |
| Generated sequences | 126,383 |
| Sequence length | 5 |
| Features per timestep | 29 |
| Shot made percentage | 45.2% |

---

# Models Implemented-

## Traditional ML Models-

- Logistic Regression
- Random Forest
- XGBoost

## Deep Learning Model-

- Attention-LSTM

## Hybrid Ensemble Models-

- XGB + LSTM Weighted Ensemble
- XGB + LSTM Stacking Ensemble
- RF + LSTM Weighted Ensemble

# Attention-LSTM Architecture-

The proposed deep learning model contains:

- LSTM Layer (128 units)
- Dropout
- Batch Normalization
- LSTM Layer (64 units)
- Bahdanau Attention Layer
- Dense Layer
- Sigmoid Output Layer

The attention mechanism helps the model focus on the most important previous shots in a sequence.

# Best Results-

| Model | Accuracy | ROC-AUC |
| XGB + LSTM Stacking| 66.31% | 0.7236 |

# Project Pipeline-

## Part 1 — Preprocessing-

The preprocessing pipeline performs:

- Dataset loading
- Cleaning
- Merging
- Feature engineering
- Rolling statistics generation
- Sequence creation
- Scaling and normalization

Run:

```bash
python part1_preprocessing.py
```

## Part 2 — Model Training

The training pipeline performs:

- Temporal train/validation/test splitting
- Baseline ML training
- Attention-LSTM training
- Hybrid ensemble learning
- Evaluation
- Visualization generation
- Model saving

Run:

```bash
python part2_training.py
```

# 📈 Generated Outputs

## Plots

- ROC Curves
- Precision-Recall Curves
- Confusion Matrices
- Attention Visualization
- Feature Importance
- Accuracy Comparison
- Metrics Heatmaps
- Training History

## Saved Models

- Attention-LSTM
- XGBoost
- Random Forest
- Hybrid Ensembles

# Project Structure-

```text
NBA-Shot-Prediction-Hybrid-LSTM-XGBoost/
│
├── part1_preprocessing.py
├── part2_training.py
├── main.tex
│
├── processed_data/
├── model_outputs/
├── datasets/
│
└── README.md
```

# Technologies Used-

- Python
- TensorFlow / Keras
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

# Research Contribution-
This work demonstrates that:

- Temporal sequence learning improves basketball shot prediction
- Attention mechanisms help identify important previous shots
- Hybrid deep learning + boosting ensembles outperform standalone models

The proposed hybrid framework achieved the best overall predictive performance.

# Future Improvements-

Possible future extensions include:

- Graph Neural Networks (GNNs)
- Transformer architectures
- Real-time prediction systems
- Player tracking coordinates
- Reinforcement learning based tactical systems

# Research Paper-
The complete IEEE-format research paper is included in this repository.

# Author-
Vivek Vijapure

# License-
This project is intended for educational and research purposes.
