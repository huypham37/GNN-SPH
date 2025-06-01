Context: The experiement is planned for my research: "How effectively can GNNs learn particle interaction dynamics in SPH simulations across different flow regimes?"

## What to collects?
- Frame per secs.
- Accumulated MAE
- avg time / step 
- training convergence time. 
- peak memory

- **Accuracy Metrics:**
  - Position prediction error (L2 norm)
  - Long-term stability (rollout error over time)
  
## What to do?
- Context: 3 different scenario in fluid simulation: Dam break, Pousielee flow and water drop. 

- For each scenario, collect data for num particle: 600, 1k, 3k, 5k. 

## Control Variables:
- Fixed time step size
- Consistent boundary conditions
- Same initial conditions across runs
- Fixed GNN architecture per comparison

## Statistical Analysis Plan (Clarified)
- **Research Question**: Does particle count (600, 1k, 3k, 5k) significantly affect prediction accuracy?
- **Method**: One-way ANOVA
  - **Independent Variable**: Particle count (4 groups)
  - **Dependent Variables**: MAE, MSE, computation time
  - **Null Hypothesis**: No difference in accuracy across particle counts
- **Follow-up**: Post-hoc tests (Tukey HSD) if significant differences found

## Potential Challenges to Address
Data Quality: Ensure SPH ground truth is numerically stable: Verified
Computational Resources: Profile memory/compute requirements
Reproducibility: Seed all random processes, document versions
Overfitting: Monitor validation performance during training
