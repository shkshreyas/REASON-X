# REASON-X: Joint Topology-Budget Allocation via Utility-Surface Regression for Test-Time Compute

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)
[![GitHub Repository](https://img.shields.io/badge/GitHub-shkshreyas%2FJTBA-blue?logo=github)](https://github.com/shkshreyas/JTBA)
[![Hugging Face Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-reason--x--ckpt-yellow)](https://huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official computational artifacts and replication suite for:  
**"REASON-X: Adaptive Test-Time Compute Allocation via Utility-Surface Regression"**

**Authors:** Shreyas Kumar (23BAI1008)  
**Supervisor:** Dr. Rishikeshan C. A.  
**Affiliation:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Chennai, Tamil Nadu, India  

---

## 1. Overview & Problem Formulation

Test-time compute in Large Language Models (LLMs) is typically scaled by increasing parallel sample count $b$ under fixed Self-Consistency (1-D budget allocation). Prior methods such as **AdaCompute** ([Zhai et al., arXiv:2604.14853](https://arxiv.org/abs/2604.14853)) restrict adaptive decisions to budget selection alone.

**REASON-X** (Joint Topology-Budget Allocation, **JTBA**) generalizes test-time compute allocation to a joint 2-D action space:
$$\mathcal{A} = \{ \text{SC}, \text{Best-of-}N, \text{Tree-of-Thought (ToT)}, \text{Forest-of-Thought (FoT)} \} \times \{ b \in \mathcal{B} \}$$

### Methodological Principles:
1. **Utility-Surface Regression ($f_\theta(x, a) \approx \mathrm{Acc}(x, a)$):** Direct regression of continuous utility surfaces across all candidate actions preserves ordinal similarity and scales the training signal by $|\mathcal{A}|\times$ compared to discrete classification.
2. **Lagrangian Dual Optimization:** Solves the constrained allocation problem under target average compute budgets $B \in \{1.5, 2.0, 3.0\}$ via dual bisection.
3. **Leak-Free Cross-Validation:** Enforces 5-fold grouped cross-validation partitioned strictly by question ID across independent random seeds.
4. **Conformal Risk Control (CRC):** Calibrates escalation confidence thresholds ($\tau = 0.95$) with prefix KV-cache reuse.

---

## 2. Repository Structure

```text
JTBA/
├── README.md                          # Comprehensive replication guide and results
├── LICENSE                            # MIT License
├── CITATION.cff                       # Machine-readable citation metadata for Zenodo
├── requirements.txt                   # Pinned Python package dependencies
├── environment.yml                    # Conda environment definition
├── .gitignore                         # Git exclusion rules
├── REASON_X_5models.ipynb             # Master replication notebook with full pre-computed outputs
├── figs/                              # Publication vector graphics
│   ├── arch1_pipeline.pdf
│   ├── arch2_actionspace.pdf
│   ├── fig_entropy.pdf
│   ├── fig_gain_ci.pdf
│   ├── fig_headroom.pdf
│   └── fig_ladder.pdf
└── archive/                           # Deprecated / raw Kaggle notebooks (excluded by git)
    ├── notebook5a08df74fa.ipynb
    └── notebookab6d1f1d7c (1).ipynb
```

---

## 3. Experimental Results Summary

Evaluated on **200 benchmark instances** (100 from **MATH-500** and 100 from **GSM8K**) across **Qwen2.5-3B-Instruct** and **Phi-3.5-mini-instruct** on dual NVIDIA Tesla T4 GPUs:

### Table 1: Accuracy Averaged over Budgets $B \in \{1.5, 2.0, 3.0\}$ (Budget-Only Space)
| Model | Fixed-$B$ | Random | S0 (AdaCompute) | S5b (Ours, 1-D) | 1-D Oracle [UB] | Gain over S0 (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B-Instruct** | 51.75% | 53.29% | 50.24% | **54.25%** | 59.34% | **+4.02 pp** |
| **Phi-3.5-mini-instruct** | 53.51% | 54.95% | 53.47% | **55.78%** | 61.45% | **+2.31 pp** |

### Statistical Significance (10,000 Paired Bootstrap Resamples)
- **Learner Gain ($S5b - S0$):**
  - `Qwen2.5-3B-Instruct`: $+4.02\text{ pp}$ [95% CI: $+2.71, +5.44$], $p = 0.0001$ (Significant)
  - `Phi-3.5-mini-instruct`: $+2.31\text{ pp}$ [95% CI: $+1.39, +3.28$], $p = 0.0001$ (Significant)
  - **Pooled:** $+3.17\text{ pp}$ [95% CI: $+2.31, +4.10$], $p = 0.0001$ (Significant)
- **Topology Gain ($S5 - S5b$):**
  - `Qwen2.5-3B-Instruct`: $+1.43\text{ pp}$ [95% CI: $+0.20, +2.84$], $p = 0.0212$ (Nominally Significant)
  - `Phi-3.5-mini-instruct`: $-0.20\text{ pp}$ [95% CI: $-1.73, +1.33$], $p = 0.7990$ (n.s.)
  - **Pooled:** $+0.62\text{ pp}$ [95% CI: $-0.42, +1.65$], $p = 0.2338$ (n.s.)

---

## 4. Experiment Checkpoints & Artifacts on Hugging Face

All intermediate generation pools, tree topologies, probe features, and utility tensors are publicly hosted on Hugging Face:  
🔗 **[huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main](https://huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main)**

### Available Artifacts:
- `ckpt/`: Serialized per-model checkpoints (`Qwen2.5-3B-Instruct_200q.pkl`, `Phi-3.5-mini-instruct_200q.pkl`).
- `figs/`: Vector PDF and high-res PNG plots (`fig3_bootstrap.png`, etc.).
- `artifacts.zip`: Compressed archive containing all empirical tensors and figures.
- `ladder.csv`, `costs.csv`, `table1_budget_only.csv`–`table5_entropy_signals.csv`: Raw tabular evaluations.
- `bootstrap_ci.csv`, `perq_accuracy.csv`: Resample distributions and per-question score matrices.

To automatically pull checkpoints locally for analysis:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="shkshreyas/reason-x-ckpt",
    repo_type="dataset",
    local_dir="./rx_out/ckpt"
)
```

---

## 5. Quick Start & Replication

### Zero-Compute Review
Open [`REASON_X_5models.ipynb`](REASON_X_5models.ipynb). All cells contain complete pre-computed outputs (execution time: 329.3 min), tables, and bootstrap statistical distributions.

### Running Analysis from Checkpoints
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shkshreyas/JTBA.git
   cd JTBA
   ```

2. **Set up Python environment:**
   ```bash
   conda env create -f environment.yml
   conda activate reason-x
   # or: pip install -r requirements.txt
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter lab REASON_X_5models.ipynb
   ```

---

## 6. Citation

```bibtex
@software{kumar2026reasonx,
  author       = {Shreyas Kumar and Rishikeshan C. A.},
  title        = {{REASON-X: Joint Topology-Budget Allocation via Utility-Surface Regression for Test-Time Compute}},
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.placeholder},
  url          = {https://github.com/shkshreyas/JTBA}
}
```

---

## 7. License
This repository is released under the [MIT License](LICENSE).
