# Methods and Experiments Enhancement Proposal

## Overview
This document proposes specific improvements to the Methods (Section 3) and Experiments sections to address anticipated reviewer concerns and strengthen the paper's contribution.

## Methods Section Improvements

### 1. Gradient Variance Analysis (High Priority)
**Current Gap**: No empirical measurement of gradient variance reduction.

**Proposed Addition**:
- Compute gradient norm variance across tokens within trajectories
- Compare: Uniform vs Surprisal vs Entropy-Reduction weighting
- Plot: Variance vs training step for each method
- Metric: Coefficient of variation (CV) of ∇log π across tokens

**Why It Matters**: Demonstrates that intrinsic weighting actually reduces variance as claimed, not just improves final accuracy.

**Implementation**:
```python
# Per-token gradient norms
grad_norms = [||∇_θ log π_θ(y_t | y_{<t})|| for t in 1..T]
within_traj_variance = var(grad_norms)  # lower is better
```

### 2. Ablation Study Checklist (Table Format)
**Proposed Table**: Systematic ablations showing:
- Uniform + no baseline (pure REINFORCE)
- Uniform + RLOO baseline
- Uniform + GRPO baseline  
- Surprisal + RLOO
- Surprisal + GRPO
- Entropy + RLOO
- Entropy + GRPO

**Metric**: Final accuracy + training steps to convergence

### 3. Sanity Check: Baseline Sensitivity
**Experiment**: Vary batch size (4, 8, 16, 32) for group baselines

**Hypothesis**: GRPO/RLOO performance degrades with small groups; intrinsic weighting more robust

**Why**: Shows token-weighting provides value even when reliable baselines are scarce

### 4. Token Weight Visualization (Figure)
**Proposed Figure**: Heatmap of token weights over reasoning trajectories

**X-axis**: Token position (t=1 to T)
**Y-axis**: Different reasoning traces  
**Color intensity**: w_t value

**Annotate**: Surprising tokens (high surprisal) vs filler tokens (low)

**Insight**: Shows credit concentration happens at semantically meaningful positions

### 5. Computational Efficiency
**Current**: "negligible overhead" claim

**Quantify**:
- Forward pass time: Uniform vs Surprisal vs Entropy
- FLOPs: Additional operations per token
- Wall-clock: End-to-end training time

**Make explicit**: No extra GPU memory (unlike value networks)

## Experiments Section Improvements

### 1. Stronger Baselines (Critical for Reviews)
**Missing**: Comparison with PPO + learned value function

**Add**:
- PPO with small value head (same compute budget)
- PPO with large value head (same parameter count)
- Show: Does critic-free actually win, or is this a fair comparison?

**Reviewer will ask**: "Why not just use PPO with smaller critic?"

### 2. Task Diversity
**Current**: Math and code reasoning

**Add**:
- **Tool use**: API call sequences (shows temporal structure matters)
- **Long-form QA**: Multi-hop reasoning (shows context length matters)

**Why**: Demonstrates method works beyond verifiable math

### 3. Failure Mode Analysis
**Section**: "When does token weighting fail?"

**Hypothesize and test**:
- Very short sequences (T < 5): All methods converge, no advantage
- Deterministic rewards: Baseline matters less
- Highly stochastic: Variance reduction crucial

**Reviewer appreciates**: Honest analysis of limitations

### 4. Hyperparameter Sensitivity
**Experiment**: Grid search over:
- Temperature (0.5, 0.7, 1.0)
- Learning rate (1e-5 to 5e-5)
- Batch size (4 to 32)

**Result**: Heatmap showing which regimes favor each method

**Why**: Shows method is practical (robust) not just optimal (brittle)

### 5. Qualitative Examples (Appendix)
**Include**: 3-4 example trajectories with:
- Full reasoning trace
- Token-level weights overlaid
- Highlight: Where did high-weight tokens land?

**Reviewer comment**: "This makes the mechanism concrete"

## Implementation Priority

### Must Have (NeurIPS survival)
1. ✅ Gradient variance measurements
2. ✅ PPO baseline comparisons
3. ✅ Computational cost quantification

### Should Have (Strong reviews)
4. Token weight visualizations
5. Failure mode analysis
6. Ablation table

### Nice to Have (Impact)
7. Additional tasks
8. Hyperparameter sensitivity
9. Qualitative examples

## Suggested Timeline

**Before paper submission**:
- Week 1-2: Gradient variance + PPO baselines ( MUST )
- Week 3: Visualizations + failure modes
- Week 4: Polish + appendix materials

**Estimated compute**: ~200 GPU hours total (V100 equivalent)

## Anticipated Reviewer Questions

### Q: "Why not use value functions on prefixes?"
**A**: Section 4 discusses — prefixes non-Markov, value estimates unreliable. Show empirical comparison!

### Q: "Is this just regularization?"
**A**: Compare to explicit entropy regularization — different mechanism, compatible combinations

### Q: "Does it scale to larger models?"
**A**: Run 7B parameter experiment (single A100) if time permits

### Q: "Why these specific weighting schemes?"
**A**: Prinicpled: surprisal = decision uncertainty; entropy-drop = commitment. Others possible (future work).
