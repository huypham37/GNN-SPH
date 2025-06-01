## Phase 4: Main Experiments

### Step 4.1: Single-Step Prediction Experiments
**For each combination** (3 scenarios × 4 particle counts = 12 experiments):

#### Training Protocol:
- **Epochs**: Max 30, early stopping (patience=10)
- **Optimizer**: AdamW with selected hyperparameters
- **Loss**: MSE on position prediction
- **Validation**: Every epoch

#### Data Collection:
- **Training metrics**:
  - Training convergence time (epochs to convergence)
  - Final training loss
  - Training time per epoch
  
- **Performance metrics**:
  - Frame per second (inference speed)
  - Average time per step
  - Peak memory usage during training/inference
  
- **Accuracy metrics**:
  - Position prediction error (L2 norm)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Relative error percentage

#### Replication:
- **5 independent runs** per configuration
- Different random seeds for reproducibility
- Report mean ± standard deviation

### Step 4.2: Long-term Stability Experiments
**For each scenario** (using best particle count from Step 4.1):

#### Multi-step Rollout:
- Predict 10, 20, 50 consecutive timesteps
- **Rollout error**: Error accumulation over time
- **Stability measure**: Energy conservation, particle distribution

#### Data Collection:
- Error growth rate over timesteps
- Qualitative assessment of flow pattern preservation
- Maximum stable prediction horizon

## Phase 5: Statistical Analysis

### Step 5.1: Particle Count Effect Analysis
- **One-way ANOVA** for each scenario:
  - **Independent variable**: Particle count (4 groups)
  - **Dependent variables**: MAE, MSE, computation time
  - **Null hypothesis**: No difference across particle counts
  
- **Post-hoc tests**: Tukey HSD if ANOVA significant
- **Effect size**: Report η² (eta-squared)

### Step 5.2: Cross-Scenario Analysis
- **Two-way ANOVA**:
  - **Factors**: Scenario type, particle count
  - **Interactions**: Scenario × particle count effects
  
- **Multiple comparisons**: Bonferroni correction
- **Confidence intervals**: 95% CI for all metrics

### Step 5.3: Computational Efficiency Analysis
- **Regression analysis**: 
  - Model: computation_time ~ particle_count + scenario
  - **Output**: Scaling relationships
  
- **Memory scaling**: 
  - Peak memory vs particle count
  - Compare GNN vs traditional SPH overhead

## Phase 6: Validation and Verification

### Step 6.1: Cross-Validation
- **k-fold cross-validation** (k=5) on combined dataset
- Verify results consistency across data splits

### Step 6.2: Generalization Testing
- Train on two scenarios, test on third
- **Purpose**: Assess cross-domain transfer capability


## Phase 7: Results Documentation

### Step 7.1: Quantitative Results
- **Tables**: Performance metrics by scenario/particle count
- **Statistical significance**: ANOVA results with p-values
- **Confidence intervals**: All reported metrics

### Step 7.2: Visualizations
- **Learning curves**: Training/validation loss over epochs
- **Scaling plots**: Performance vs particle count
- **Error distributions**: Histogram of prediction errors
- **Trajectory visualizations**: Qualitative flow pattern assessment

### Step 7.3: Computational Analysis
- **Memory usage plots**: Peak memory vs particle count
- **Timing analysis**: Training time vs dataset size
- **Efficiency comparison**: GNN vs traditional SPH

