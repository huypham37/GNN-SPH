# GNN-SPH: Graph Neural Networks for Smoothed Particle Hydrodynamics

This repository contains a research project that applies Graph Neural Networks (GNNs) to learn particle interaction dynamics in Smoothed Particle Hydrodynamics (SPH) simulations across different flow regimes.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Latest-orange.svg)](https://pytorch-geometric.readthedocs.io)

## 🔬 Research Overview

**Research Question**: How effectively can Graph Neural Networks learn particle interaction dynamics in SPH simulations across different flow regimes?

This project investigates the capability of GNNs to predict next-frame particle positions in fluid dynamics simulations, evaluating performance across three distinct flow scenarios with varying particle counts.

### Key Research Contributions

- **Novel GNN Architecture**: Custom message-passing network with skip connections optimized for SPH dynamics
- **Multi-Scale Analysis**: Evaluation across 600-5K particles to study computational scaling
- **Cross-Domain Validation**: Performance analysis on Dam Break, Lid-Driven Cavity, and Reverse Poiseuille Flow scenarios
- **Comprehensive Benchmarking**: Accuracy, memory usage, and inference speed analysis

## 🏗️ Project Structure

```
GNN-SPH/
├── datasets/                     # SPH simulation datasets
│   ├── 2D_DAM_5740_20kevery100/  # Dam break scenario (5740 particles)
│   ├── 2D_LDC_2708_10kevery100/  # Lid-driven cavity (2708 particles)
│   └── 2D_RPF_3200_20kevery100/  # Reverse Poiseuille flow (3200 particles)
├── models/                       # Trained model checkpoints
│   ├── dam_gnn_model_15.pth     # Dam break model (15 layers)
│   ├── ldc_gnn_model_15.pth     # Lid-driven cavity model
│   └── rpf_gnn_model.pth        # Reverse Poiseuille flow model
├── notebooks/                    # Training and analysis notebooks
├── stats/                        # Performance statistics and results
├── inference_pipeline.ipynb      # Main inference and evaluation pipeline
└── download_data.sh              # Dataset download script
```

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install h5py matplotlib numpy psutil
```

### Download Datasets

```bash
# Make script executable
chmod +x download_data.sh

# Download specific dataset
bash download_data.sh dam_2d datasets/

# Download all datasets
bash download_data.sh all datasets/
```

Available datasets:
- `dam_2d`: 2D Dam Break (5740 particles)
- `ldc_2d`: 2D Lid-Driven Cavity (2708 particles) 
- `rpf_2d`: 2D Reverse Poiseuille Flow (3200 particles)

### Quick Inference Example

```python
import torch
from inference_pipeline import load_model, load_dataset, collect_episode_inference_statistics

# Load pre-trained model and dataset
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = load_model('models/dam_gnn_model_15.pth', device)
positions, particle_types = load_dataset('datasets/2D_DAM_5740_20kevery100/test.h5')

# Run inference with statistics collection
stats = collect_episode_inference_statistics(
    model, positions, particle_types, 
    episode_idx=0, device=device, max_particles=2000
)
```

## 🧠 Model Architecture

### ParticleDynamicsGNN

Our custom GNN architecture is specifically designed for SPH particle dynamics:

```
Input Features (5D):
├── Position (x, y)
├── Particle Type (one-hot: fluid/boundary)
└── Timestep (normalized)

Architecture:
├── Input Embedding: Linear(5) → ReLU → Dropout → 128
├── Message Passing Layers (15 layers):
│   ├── Message MLP: Concat(x_i, x_j) → Linear → ReLU → Linear
│   ├── Update MLP: Concat(node, messages) → Linear → ReLU → Linear
│   └── Skip Connections: Every 2 layers
└── Output Layer: Linear(128) → ReLU → Linear(2) [Δx, Δy]
```

### Key Features

- **Message Passing**: Custom neighbor communication with spatial graph construction
- **Skip Connections**: Residual connections every 2 layers for gradient flow
- **Spatial Partitioning**: Efficient neighbor finding using grid-based spatial partitioning
- **Particle Subsampling**: Intelligent sampling preserving all boundary particles

## 📊 Performance Results

### Accuracy Metrics (Dam Break Scenario)

```
Average MAE:      0.350830
Average MSE:      0.350700
RMSE:             0.592199
Max Error:        2.048591
```

### Computational Performance

```
Inference Speed:  6.03 FPS
Avg Time/Step:    165.917 ms
Peak Memory:      999.77 MB
Memory Overhead:  636.52 MB
```

### Scaling Analysis

| Particle Count | MAE | Inference Time (ms) | Memory (MB) |
|---------------|-----|-------------------|-------------|
| 1K            | 0.28 | 95.2             | 485         |
| 2K            | 0.35 | 165.9            | 636         |
| 3K            | 0.41 | 245.1            | 812         |
| 5K            | 0.52 | 398.7            | 1204        |

## 🔬 Experiments and Analysis

### Experimental Design

Our comprehensive evaluation covers:

1. **Single-Step Prediction**: Next-frame position prediction accuracy
2. **Multi-Scale Analysis**: Performance across 600, 1K, 3K, 5K particles
3. **Cross-Scenario Validation**: Dam Break, Lid-Driven Cavity, Reverse Poiseuille Flow
4. **Statistical Analysis**: ANOVA and post-hoc tests for significance

### Datasets

- **Dam Break**: Fluid collapse simulation (5740 particles)
- **Lid-Driven Cavity**: Steady flow in closed cavity (2708 particles)  
- **Reverse Poiseuille Flow**: Pressure-driven flow (3200 particles)

Each dataset includes:
- Training/validation/test splits
- 400+ timesteps per episode
- Multiple episodes for statistical robustness

## 📈 Visualization and Analysis

The repository includes comprehensive visualization tools:

```python
# Visualize single prediction
visualize_single_prediction_fixed(model, test_loader, device, sample_idx=0)

# Create prediction animations
create_animation_from_predictions(model, positions, particle_types, frames=60)

# Collect detailed statistics
episode_stats = collect_episode_inference_statistics(
    model, positions, particle_types, episode_idx=0, 
    max_particles=2000, max_timesteps=400
)
```

### Available Analyses

- **Learning Curves**: Training/validation loss progression
- **Error Distributions**: Prediction error histograms
- **Scaling Plots**: Performance vs. particle count
- **Memory Profiling**: Peak memory usage tracking
- **Trajectory Visualization**: Qualitative flow pattern assessment

## 🛠️ Advanced Usage

### Custom Training

```python
# Train your own model
from notebooks.gnn_training_pipeline import train_model

model = train_model(
    dataset_path='datasets/2D_DAM_5740_20kevery100/',
    hidden_channels=128,
    num_layers=15,
    dropout=0.15,
    max_epochs=30
)
```

### Batch Inference

```python
# Process multiple episodes
for episode_idx in range(num_episodes):
    stats = collect_episode_inference_statistics(
        model, positions, particle_types, episode_idx,
        device=device, max_particles=2000
    )
    save_statistics(stats, f'episode_{episode_idx}_stats.csv')
```

### Custom Scenarios

```python
# Adapt for new SPH scenarios
def prepare_custom_data(positions, particle_types, timestep_idx):
    # Custom feature engineering
    type_one_hot = F.one_hot(particle_types, num_classes=2).float()
    timestep_feature = torch.full((len(positions), 1), timestep_idx / 100.0)
    node_features = torch.cat([positions, type_one_hot, timestep_feature], dim=1)
    
    # Build spatial graph
    edge_index = build_graph(positions, radius=0.08)
    return Data(x=node_features, edge_index=edge_index)
```

