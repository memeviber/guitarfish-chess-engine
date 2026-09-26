# Release Notes: Model 5.3.29

Guitarfish release **5.3.29** updates the engine's custom INT8 NNUE evaluation model, superseding checkpoint **5.3.28**. This release concludes the scheduled training pipeline on the 17.6M quiet-position dataset, doubling the training duration and activating **Stochastic Weight Averaging (SWA)** for enhanced generalization and evaluation stability.

---

## Model Comparison Matrix

| Parameter / Metric | Model 5.3.28 (Previous) | Model 5.3.29 (Current) | Change / Impact |
| :--- | :--- | :--- | :--- |
| **Training Duration** | 13 Epochs | **25 Epochs** | Full convergence across all 17.6M FENs |
| **Weight Optimization** | Standard AdamW Checkpoint | **SWA (Stochastic Weight Averaging)** | Smoothed loss landscape, wider local minima |
| **Dataset** | 17.6M Quiet FENs | 17.6M Quiet FENs | Unchanged (identical baseline distribution) |
| **Dual Loss ($0.5\text{BCE} + 0.5\text{Huber}$)** | Intermediate / Early-stop | **Fully Converged** | Substantially reduced validation loss |
| **Evaluation Jitter** | Moderate | **Low** | Smoother transitions between adjacent plies |
| **Packaging** | `.gm` binary bundle | `.gm` binary bundle | Drop-in replacement (backward-compatible) |

---

## Technical Deep Dive

### 1. Stochastic Weight Averaging (SWA) Integration
Model 5.3.28 represented an intermediate checkpoint taken before the SWA schedule was engaged. In contrast, 5.3.29 averages network weights across multiple late-stage training cycles along the optimization trajectory:

- **Flat Minima Convergence:** Standard SGD/Adam optimizers often settle into sharp local minima that overfit the training corpus. SWA guides the parameter set toward flatter regions of the loss landscape, significantly boosting out-of-sample generalization in unfamiliar, out-of-book middlegames.
- **Evaluation Noise Reduction:** By averaging parameter states, SWA suppresses gradient variance. This mitigates evaluation "jitter"—minor, erratic score swings between structurally identical positions.

### 2. Extended Training Horizon (13 $\rightarrow$ 25 Epochs)
- **Centipawn Granularity:** Doubling the training duration allowed the Huber component of the loss function to penalize subtle positional imbalances more effectively, yielding sharper piece-square and combat-map weighting.
- **WDL Confidence Calibration:** The Binary Cross-Entropy (BCE) term reached asymptotic stability, producing better-calibrated winning probabilities in non-tactical, strategic endgames.

---

## Impact on Search Performance

While the underlying search algorithm remains unchanged, the refined evaluation model directly benefits tree traversal efficiency:

* **Higher Pruning Fidelity:** Reduced evaluation jitter translates to more consistent static scores, cutting down on faulty early cutoffs in **Reverse Futility Pruning (RFP)** and **Futility Pruning (FP)**.
* **Safer Reductions:** More dependable positional scoring prevents **Late Move Reductions (LMR)** from under-searching critical quiet refutations.
* **Endgame Conversion:** Superior pawn-structure and king-safety feature resolution prevents positional drift in technical endings.

---

## Deployment & Verification

Model 5.3.29 is packed directly into the default `.gm` container. No runtime engine flag changes or dependency upgrades are required.

```bash
# Verify model integrity via UCI info handshake
python guitarfish.py guitarfish.gm book.bin
uci
# Check initialization log for model identifier: 5.3.29
```