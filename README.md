# Learning Where to Spend Test-Time Compute: A Utility-Surface Allocator, and What Choosing the Reasoning Topology Is Worth

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23360219.svg)](https://doi.org/10.5281/zenodo.23360219)
[![GitHub Repository](https://img.shields.io/badge/GitHub-shkshreyas%2FREASON--X-blue?logo=github)](https://github.com/shkshreyas/REASON-X)
[![Hugging Face Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-reason--x--ckpt-yellow)](https://huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official computational artifacts and replication suite for:  
**"Learning where to spend test-time compute: a utility-surface allocator, and what choosing the reasoning topology is worth"**

**Student Author:** Shreyas Kumar (23BAI1008)  
**Corresponding Author & Supervisor:** Dr. Rishikeshan C. A. (`rishikeshan.ca@vit.ac.in`)  
**Affiliation:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Chennai 600127, Tamil Nadu, India  

---

## 1. Overview & Contributions

Spending more inference compute on difficult instances improves the reasoning accuracy of language models. Recent baseline work ([Zhai et al., arXiv:2604.14853](https://arxiv.org/abs/2604.14853), *AdaCompute*) determines the per-question budget via Lagrangian relaxation and trains a classifier to imitate the oracle budget under a fixed Self-Consistency engine (1-D budget allocation).

This work (**JTBA: Joint Topology-Budget Allocation**, also referred to as **REASON-X**) investigates two core extensions:

1. **Utility-Surface Regression vs. Discrete Classification:**  
   Instead of collapsing training data into single discrete argmax labels (which causes label collapse and discards ordinal similarity), we fit a regressor $g_\theta(x, \alpha)$ predicting $\text{Acc}(x, \alpha)$ for every action $\alpha = (a, b)$ across all $N \times |\mathcal{A}|$ measured evaluations. The Lagrangian dual is then solved directly on the predicted surface.
2. **Joint Topology-Budget Action Space:**  
   We generalize the action space from simple budget scaling ($b \in \mathcal{B}$) to a 2-D joint action set combining reasoning topologies and sample budgets:
   $$\mathcal{A} = \{\text{SC}\} \times \{1, 2, 4, 8, 16\} \cup \{\text{BoN}\} \times \{2, 4, 8, 16\} \cup \{(\text{ToT}, 4), (\text{ToT}, 8), (\text{FoT}, 8)\}, \quad |\mathcal{A}| = 12$$
3. **Cost-Monotonicity Guarantee (Proposition 1):**  
   We formally prove that for any finite action set $\mathcal{A}$ with measured token costs $C(\alpha)$, the induced average cost $\bar{C}(\lambda)$ is strictly non-increasing in the dual multiplier $\lambda$ under any deterministic tie-breaking rule preferring cheaper actions. Dual binary search reaches any target budget $B$ in $O(\log(1/\epsilon))$ iterations.
4. **Empirical Evaluation Across Five Solvers:**  
   Evaluated on 200 mathematical reasoning questions (130 from **MATH-500**, 70 from **GSM8K**) across **five solvers from three model families** spanning 1.5B to 3.8B parameters.

---

## 2. Experimental Results (Full 5-Model Evaluation)

### Table 1: Budget-Only Action Set Performance ($B \in \{1.5, 2.0, 3.0\}$, Out-of-Fold across 5 Seeds)
*Comparing the surface allocator ($S5b$) against the reproduced classifier ($S0$), the budget-matched random baseline, and the 1-D oracle upper bound.*

| Solver | Parameters | Fixed-b | Random | S0 (AdaCompute) | S5b (Ours) | 1-D Oracle [UB] | Gain over S0 (pp) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 1.5B | 0.3633 | 0.3820 | 0.3673 | **0.4036** | 0.4632 | **+3.63** [2.50, 4.88] |
| **Qwen2.5-Math-1.5B** | 1.5B | 0.4722 | 0.4854 | 0.4629 | **0.5088** | 0.5588 | **+4.59** [3.22, 6.11] |
| **SmolLM2-1.7B** | 1.7B | 0.2422 | 0.2517 | 0.2479 | **0.2637** | 0.3506 | **+1.57** [0.53, 2.65] |
| **Qwen2.5-3B** | 3.0B | 0.5175 | 0.5329 | 0.5024 | **0.5425** | 0.5934 | **+4.02** [2.71, 5.44] |
| **Phi-3.5-mini** | 3.8B | 0.5351 | 0.5495 | 0.5347 | **0.5578** | 0.6145 | **+2.31** [1.39, 3.28] |
| **Pooled, 1.5–1.7B (3)** | 1.5–1.7B | — | — | — | — | — | **+3.26** [2.52, 4.03] |
| **Pooled, 3–3.8B (2)** | 3–3.8B | — | — | — | — | — | **+3.17** [2.31, 4.10] |

*Key finding:* The utility-surface allocator ($S5b$) improves on the classifier ($S0$) across **all five solvers**, with 95% paired-bootstrap confidence intervals strictly excluding zero in every case.

---

### Table 2: Accuracy Progression by Methodological Stage & Out-of-Fold Surface $R^2$
*Stages $S1$ to $S5$ operate on the joint action set; $S0$ and $S5b$ operate on the budget-only set.*

| Solver | S0 (1-D clf) | S1 (2-D clf) | S2 (Surface Reg) | S3 (+Semantic H) | S5 (+Probes) | S5b (1-D ablation) | $R^2$ (S2 / S3 / S5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 0.3673 | 0.3849 | 0.3906 | 0.3952 | 0.3948 | 0.4036 | 0.27 / 0.49 / 0.52 |
| **Qwen2.5-Math-1.5B** | 0.4629 | 0.4678 | 0.5140 | 0.5188 | 0.5199 | 0.5088 | 0.48 / 0.58 / 0.60 |
| **SmolLM2-1.7B** | 0.2479 | 0.2519 | 0.2611 | 0.2656 | 0.2643 | 0.2637 | 0.08 / 0.27 / 0.29 |
| **Qwen2.5-3B** | 0.5024 | 0.5119 | 0.5497 | 0.5571 | 0.5569 | 0.5425 | 0.46 / 0.54 / 0.63 |
| **Phi-3.5-mini** | 0.5347 | 0.5334 | 0.5602 | 0.5579 | 0.5559 | 0.5578 | 0.21 / 0.26 / 0.46 |

*Key finding:* Replacing discrete classification with utility-surface regression ($S1 \to S2$) contributes the vast majority of accuracy gains (0.57 to 4.62 pp). Semantic entropy ($S2 \to S3$) makes the surface predictable, substantially boosting out-of-fold $R^2$ across all solvers.

---

### Table 3: Oracle Accuracy and Headroom by Target Budget
*Oracle headroom represents the theoretical maximum performance gain achievable by dynamically selecting reasoning topologies over budget scaling alone.*

| Solver | 1-D Oracle | 2-D Oracle | Headroom ($B=1.5$) | Headroom ($B=2.0$) | Headroom ($B=3.0$) | Mean Headroom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 0.4632 | 0.5475 | +8.63 pp | +8.50 pp | +8.16 pp | **+8.43 pp** |
| **Qwen2.5-Math-1.5B** | 0.5588 | 0.5775 | +1.19 pp | +2.16 pp | +2.28 pp | **+1.88 pp** |
| **SmolLM2-1.7B** | 0.3506 | 0.3566 | +0.75 pp | +0.47 pp | +0.59 pp | **+0.60 pp** |
| **Qwen2.5-3B** | 0.5934 | 0.6297 | +3.33 pp | +4.29 pp | +3.26 pp | **+3.63 pp** |
| **Phi-3.5-mini** | 0.6145 | 0.6584 | +4.55 pp | +4.98 pp | +3.64 pp | **+4.39 pp** |

---

### Table 4: Surviving Actions, Verifier Discrimination, and Learned Topology Gain
*Pareto filtering determines which non-dominated actions survive. Learned topology gain measures $S5 - S5b$.*

| Solver | Surviving Tree Actions | Verifier AUROC | Mean Oracle Headroom | Learned Topology Gain ($S5 - S5b$) [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | ToT@4 | 0.806 | +8.43 pp | -0.88 [-4.04, 2.21] |
| **Qwen2.5-Math-1.5B** | *None* (dominated) | 0.922 | +1.88 pp | +1.11 [0.01, 2.29] |
| **SmolLM2-1.7B** | *None* (dominated) | 0.529 | +0.60 pp | +0.06 [-0.29, 0.41] |
| **Qwen2.5-3B** | ToT@4, ToT@8 | 0.933 | +3.63 pp | **+1.43** [0.20, 2.84] |
| **Phi-3.5-mini** | ToT@4 | 0.866 | +4.39 pp | -0.20 [-1.73, 1.33] |
| **Pooled, 1.5–1.7B (3)** | — | — | — | +0.10 [-0.98, 1.20] |
| **Pooled, 3–3.8B (2)** | — | — | — | +0.62 [-0.42, 1.65] |

*Key finding:* Topology headroom is substantial (3.6 to 8.4 pp) only when tree search survives Pareto filtering. However, current lightweight features recover little of this headroom under learned policies; the topology gain is nominally significant only on Qwen2.5-3B (+1.43 pp), but does not survive Bonferroni correction across five comparisons.

---

### Table 5: Feature Correlations with Single-Pass Solvability & Surface $R^2$
*Correlation ($|r|$) with single-pass accuracy $\text{Acc}(x, \text{SC}, 1)$ and out-of-fold utility-surface fit.*

| Solver | Token Entropy ($r$) | Semantic Entropy ($r$) | Text Features $R^2$ | $+H_{\text{sem}}$ $R^2$ | $+$Probe Features $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | +0.100 | **-0.705** | 0.269 | 0.494 | **0.522** |
| **Qwen2.5-Math-1.5B** | +0.032 | **-0.789** | 0.479 | 0.580 | **0.603** |
| **SmolLM2-1.7B** | -0.349 | **-0.394** | 0.084 | 0.274 | **0.289** |
| **Qwen2.5-3B** | -0.161 | **-0.734** | 0.463 | 0.539 | **0.631** |
| **Phi-3.5-mini** | -0.391 | **-0.639** | 0.210 | 0.265 | **0.465** |

*Key finding:* Semantic entropy ($H_{\text{sem}}$) consistently demonstrates strong negative correlation with instance solvability (-0.39 to -0.79), whereas token entropy correlates weakly (-0.39 to +0.10). Adding $H_{\text{sem}}$ elevates $R^2$ far more than all sixteen surface text features combined.

---

## 3. Publication Figures (Included in `figs/`)

The vector graphics matching the paper are included in the [`figs/`](figs/) directory:

- **[`figs/arch1_pipeline.pdf`](figs/arch1_pipeline.pdf)** (Figure 1): Complete architectural overview of the offline utility tensor generation, surface regressor fitting, online probe extraction, and dual bisection allocator.
- **[`figs/arch2_actionspace.pdf`](figs/arch2_actionspace.pdf)** (Figure 2): Diagram contrasting the 1-D budget allocation space ($|\mathcal{A}|=5$) with the 2-D joint topology-budget space ($|\mathcal{A}|=12$).
- **[`figs/fig_gain_ci.pdf`](figs/fig_gain_ci.pdf)** (Figure 3): Two-panel paired bootstrap 95% confidence intervals across all five solvers and pooled groups (Allocator Gain vs. Topology Gain).
- **[`figs/fig_ladder.pdf`](figs/fig_ladder.pdf)** (Figure 4): Multi-panel plots illustrating stage progression ($S0 \to S5$) against the 1-D and 2-D empirical oracles.
- **[`figs/fig_headroom.pdf`](figs/fig_headroom.pdf)** (Figure 5): Oracle headroom comparison across solvers and target budgets ($B \in \{1.5, 2.0, 3.0\}$).
- **[`figs/fig_entropy.pdf`](figs/fig_entropy.pdf)** (Figure 6): Left: correlation of token vs. semantic entropy with solvability. Right: out-of-fold surface $R^2$ fit progression across feature sets.

---

## 4. Repository Structure

```text
REASON-X/
├── README.md                  # Comprehensive empirical report & replication documentation
├── LICENSE                    # MIT License
├── CITATION.cff               # Citation File Format 1.2.0 for Zenodo & GitHub
├── requirements.txt           # Python package dependencies
├── environment.yml            # Conda environment specification
├── .gitignore                 # Git ignore rules (excludes archive/ and local checkpoints)
├── REASON_X_5models.ipynb     # Master experiment notebook with complete pre-computed outputs
└── figs/                      # Publication-ready vector graphics (PDFs)
    ├── arch1_pipeline.pdf
    ├── arch2_actionspace.pdf
    ├── fig_entropy.pdf
    ├── fig_gain_ci.pdf
    ├── fig_headroom.pdf
    └── fig_ladder.pdf
```

---

## 5. Experiment Checkpoints & Data Availability

- **Hugging Face Hub Repository:**  
  🔗 **[huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main](https://huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main)**
- **Retained Checkpoints:**  
  Full empirical utility tensors, tree traces, probe caches, and per-question accuracy matrices for the two larger models (**Qwen2.5-3B-Instruct** and **Phi-3.5-mini-instruct**) are archived on Hugging Face (`ckpt/`).
- **Reproducibility of 1.5–1.7B Solvers:**  
  As reported in the paper (Section 8.5 & Data Availability), checkpoints for the three smaller solvers (`Qwen2.5-1.5B`, `Qwen2.5-Math-1.5B`, `SmolLM2-1.7B`) were not retained due to storage constraints; they can be regenerated in ~2 hours each on a single T4 GPU using [`REASON_X_5models.ipynb`](REASON_X_5models.ipynb) with the fixed seed (`CFG.seed = 42`).
- **Tabular Artifacts:**  
  All output CSVs (`table1_budget_only.csv` through `table5_entropy_signals.csv`, `ladder.csv`, `costs.csv`, `bootstrap_ci.csv`, `perq_accuracy.csv`) and `artifacts.zip` are directly downloadable from the Hugging Face repository.

---

## 6. Replication Instructions

### A. Inspecting Existing Results (No Compute Required)
The master notebook [`REASON_X_5models.ipynb`](REASON_X_5models.ipynb) contains complete pre-computed execution outputs, logs, tables, and bootstrap statistical distributions.

### B. Environment Setup & Re-Running Analysis
```bash
# Clone the repository
git clone https://github.com/shkshreyas/REASON-X.git
cd REASON-X

# Create and activate conda environment
conda env create -f environment.yml
conda activate reason-x

# Or install via pip
pip install -r requirements.txt

# (Optional) Download checkpoints from Hugging Face
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='shkshreyas/reason-x-ckpt', repo_type='dataset', local_dir='./rx_out/ckpt')"

# Launch Jupyter
jupyter lab REASON_X_5models.ipynb
```

---

## 7. Citation

If you use this codebase, methodology, or experimental findings, please cite:

```bibtex
@article{kumar2026learning,
  title   = {Learning where to spend test-time compute: a utility-surface allocator, and what choosing the reasoning topology is worth},
  author  = {Kumar, Shreyas and Rishikeshan, C. A.},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/shkshreyas/REASON-X}
}

@software{kumar2026jtba_code,
  author    = {Shreyas Kumar and Rishikeshan C. A.},
  title     = {{JTBA: Joint Topology-Budget Allocation via Utility-Surface Regression}},
  month     = oct,
  year      = 2026,
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23360219},
  url       = {https://github.com/shkshreyas/REASON-X}
}
```

---

## 8. License
This repository is released under the [MIT License](LICENSE).
