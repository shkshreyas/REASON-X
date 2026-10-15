# Learning Where to Spend Test-Time Compute in Language Models: A Utility-Surface Allocator and the Value of Choosing the Reasoning Topology

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23360219-blue.svg?logo=zenodo)](https://doi.org/10.5281/zenodo.23360219)
[![GitHub Repository](https://img.shields.io/badge/GitHub-shkshreyas%2FREASON--X-blue?logo=github)](https://github.com/shkshreyas/REASON-X)
[![Hugging Face Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-reason--x--ckpt-yellow)](https://huggingface.co/datasets/shkshreyas/reason-x-ckpt/tree/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official computational artifacts, replication suite, and empirical results for:  
**"Learning where to spend test-time compute in language models: a utility-surface allocator and the value of choosing the reasoning topology"**

**Authors:** Shreyas Kumar, Rishikeshan C. A.*  
**Affiliation:** School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Chennai 600127, Tamil Nadu, India  
**Correspondence:** `rishikeshan.ca@vit.ac.in` | `shreyas.kumar2023@vitstudent.ac.in`

---

## Abstract

A language model answers reasoning questions more accurately when it is allowed extra inference-time computation, but every additional sample costs more and most questions need little. A recent method sets each question's sample budget by solving a budget-constrained problem through its Lagrangian dual and trains a classifier to copy the resulting budgets. We revise this decision system in two ways and evaluate each revision separately. First, a regressor predicts the accuracy of every available action, using semantic entropy from a short probe as a difficulty signal, and the same dual problem is solved on its predictions. Second, the system decides the form of computation (majority voting, verifier reranking, tree search or forest search) along with its amount, and we prove that the dual solver's monotonicity guarantee extends to any finite set of actions. On 200 MATH-500 and GSM8K questions answered by six open models from three families (1.5B to 7B parameters), the regressor beats the classifier in the classifier's own setting on every model, with all six paired-bootstrap intervals above zero. Choosing the topology raises the best attainable accuracy by 3.6 to 8.4 points on the four models that retain a tree-search action after Pareto filtering, against 0.6 to 1.9 points for the other two. A learned policy captures most of that headroom only on the largest model, recovering 4.7 of 6.0 points (95% interval 2.1 to 7.5), where its utility predictions are most accurate. A distribution-free certificate for verifier-accepted answers fails on all six models.

**Keywords:** Test-time compute, Resource allocation, Large language models, Lagrangian relaxation, Semantic entropy, Decision support

---

## 1. Key Contributions

1. **Utility-surface allocator:** A regressor that predicts the accuracy of every action and solves the dual problem on the prediction. With semantic entropy among its features, it outperforms the classifier of Zhai et al. (AdaCompute) in that classifier's budget-only setting on all six models studied, by 1.6 to 4.6 accuracy points, each with a 95% paired-bootstrap interval strictly above zero.
2. **Joint topology-budget allocation:** An allocation formulation over joint topology and budget choices with costs taken from generated-token counts, with a proof (Proposition 1) that dual bisection on the multiplier stays valid however the finite action set is built.
3. **Value of topology choice:** Given true accuracies, allowing search and reranking lifts accuracy by 3.6 to 8.4 points on the four models whose search actions survive Pareto filtering, and by only 0.6 to 1.9 points where none survives.
4. **Learned headroom realization at 7B scale:** On five smaller models the learned gain from the topology dimension is small. On **Qwen2.5-7B-Instruct**, the model whose utility surface is predicted best, the learned policy recovers **4.7 of 6.0 available points** (95% interval 2.1 to 7.5, $p < 0.001$), which remains significant after Bonferroni correction.
5. **Behavioral difficulty signals:** Semantic entropy tracks single-sample accuracy far more closely than token entropy ($r = -0.64$ to $-0.79$), and on its own improves the surface fit more than sixteen surface text statistics combined.
6. **Probe reuse & conformal risk certification:** Continuing the difficulty probe into the final answer saves 0.75 to 0.94 sample-equivalents per question. A Learn-then-Test certificate on verifier-accepted answers fails to meet risk targets across all six models.

---

## 2. Experimental Setup

- **Question Benchmark:** 200 mathematical reasoning questions (130 from **MATH-500**, 70 from **GSM8K**) sampled once with a fixed seed (`CFG.seed = 42`). Every model answers the identical 200 questions.
- **Model Suite (Table 1 of Manuscript):**

| Model | Family | Parameters | Role in Comparison |
| :--- | :---: | :---: | :--- |
| **Qwen2.5-1.5B-Instruct** | Qwen2.5 | 1.5B | General, smallest |
| **Qwen2.5-Math-1.5B-Instruct** | Qwen2.5 | 1.5B | Specialised for mathematics |
| **SmolLM2-1.7B-Instruct** | SmolLM2 | 1.7B | Weak self-verifier |
| **Qwen2.5-3B-Instruct** | Qwen2.5 | 3.0B | General, middle size |
| **Phi-3.5-mini-instruct** | Phi-3.5 | 3.8B | Different model family |
| **Qwen2.5-7B-Instruct** | Qwen2.5 | 7.0B | General, largest |

- **Action Space ($|\mathcal{A}| = 12$):**
  $$\mathcal{A} = \{\text{SC}\} \times \{1, 2, 4, 8, 16\} \cup \{\text{BoN}\} \times \{2, 4, 8, 16\} \cup \{(\text{ToT}, 4), (\text{ToT}, 8), (\text{FoT}, 8)\}$$
- **Methodological Stages:**
  - **S0:** Baseline classifier imitating oracle budget on budget-only actions (reproducing Zhai et al.).
  - **S1:** Discrete classifier on the joint topology-budget action space.
  - **S2:** Utility-surface regression on question text features and action encoding.
  - **S3:** Surface regression with semantic entropy ($H_{\text{sem}}$) from a 64-token probe.
  - **S4:** Full surface regression including four probe features (verifier spread, answer agreement, verifier contradiction flag, distinct step share).
  - **S4b:** S4 restricted to budget-only actions (isolates learner improvement from topology choice).

---

## 3. Empirical Results

### Table 2: Budget-Only Action Set Performance ($B \in \{1.5, 2.0, 3.0\}$, Out-of-Fold across Seeds)
*Comparing the surface allocator ($S4b$) against the reproduced classifier ($S0$), the budget-matched random baseline, and the 1-D oracle upper bound. The last column is $S4b - S0$ in accuracy points with 95% paired-bootstrap intervals.*

| Model | Fixed | Random | S0 | S4b (Ours) | 1-D Oracle | Gain (pp) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 0.3633 | 0.3820 | 0.3673 | **0.4036** | 0.4632 | **+3.63** [2.50, 4.88] |
| **Qwen2.5-Math-1.5B** | 0.4722 | 0.4854 | 0.4629 | **0.5088** | 0.5588 | **+4.59** [3.22, 6.11] |
| **SmolLM2-1.7B** | 0.2422 | 0.2517 | 0.2479 | **0.2637** | 0.3506 | **+1.57** [0.53, 2.65] |
| **Qwen2.5-3B** | 0.5175 | 0.5329 | 0.5024 | **0.5425** | 0.5934 | **+4.02** [2.71, 5.44] |
| **Phi-3.5-mini** | 0.5351 | 0.5495 | 0.5347 | **0.5578** | 0.6145 | **+2.31** [1.39, 3.28] |
| **Qwen2.5-7B** | 0.5576 | 0.5700 | 0.5386 | **0.5845** | 0.6317 | **+4.58** [3.19, 6.10] |
| **Pooled, 1.5–1.7B (3)** | — | — | — | — | — | **+3.26** [2.52, 4.03] |
| **Pooled, 3–7B (3)** | — | — | — | — | — | **+3.64** [2.77, 4.58] |

---

### Table 3: Accuracy Progression by Stage & Out-of-Fold Surface $R^2$
*Stages S1 to S4 operate on the joint action set; S0 and S4b use budgets only. Out-of-fold $R^2$ reported for S2, S3, and S4.*

| Model | S0 | S1 | S2 | S3 | S4 | S4b | $R^2$ (S2 / S3 / S4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 0.3673 | 0.3849 | 0.3906 | 0.3952 | 0.3948 | 0.4036 | 0.27 / 0.49 / 0.52 |
| **Qwen2.5-Math-1.5B** | 0.4629 | 0.4678 | 0.5140 | 0.5188 | 0.5199 | 0.5088 | 0.48 / 0.58 / 0.60 |
| **SmolLM2-1.7B** | 0.2479 | 0.2519 | 0.2611 | 0.2656 | 0.2643 | 0.2637 | 0.08 / 0.27 / 0.29 |
| **Qwen2.5-3B** | 0.5024 | 0.5119 | 0.5497 | 0.5571 | 0.5569 | 0.5425 | 0.46 / 0.54 / 0.63 |
| **Phi-3.5-mini** | 0.5347 | 0.5334 | 0.5602 | 0.5579 | 0.5559 | 0.5578 | 0.21 / 0.26 / 0.46 |
| **Qwen2.5-7B** | 0.5386 | 0.5603 | 0.6354 | 0.6322 | 0.6314 | 0.5845 | 0.51 / 0.70 / 0.71 |

---

### Table 4: Lagrangian Oracle Accuracy and Headroom Across Budgets
*1-D oracle (budget only) vs. 2-D oracle (joint topology-budget set) and headroom in accuracy points. Realised mean costs reported for models with token logging.*

| Model | 1-D Oracle | 2-D Oracle | $B = 1.5$ | $B = 2.0$ | $B = 3.0$ | Mean Headroom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | 0.4632 | 0.5475 | 8.63 | 8.50 | 8.16 | **8.43** |
| **Qwen2.5-Math-1.5B** | 0.5588 | 0.5775 | 1.19 | 2.16 | 2.28 | **1.88** |
| **SmolLM2-1.7B** | 0.3506 | 0.3566 | 0.75 | 0.47 | 0.59 | **0.60** |
| **Qwen2.5-3B** | 0.5934 | 0.6297 | 3.33 | 4.29 | 3.26 | **3.63** |
| **Phi-3.5-mini** | 0.6145 | 0.6584 | 4.55 | 4.98 | 3.64 | **4.39** |
| **Qwen2.5-7B** | 0.6317 | 0.6915 | 6.59 | 7.67 | 3.70 | **5.99** |

*Realised mean cost, 1-D / 2-D (sample-equivalents):*
- **Qwen2.5-3B:** 2.14 / 2.13 (at $B=1.5$: 1.50/1.49; $B=2.0$: 1.98/1.98; $B=3.0$: 2.95/2.93)
- **Phi-3.5-mini:** 2.16 / 2.04 (at $B=1.5$: 1.50/1.50; $B=2.0$: 2.00/1.99; $B=3.0$: 2.99/2.65)
- **Qwen2.5-7B:** 2.16 / 1.86 (at $B=1.5$: 1.50/1.48; $B=2.0$: 1.98/1.99; $B=3.0$: 3.00/2.09)

---

### Table 5: Surviving Search Actions, Verifier Quality, and Learned Topology Gain
*Pareto filtering determines which tree-search actions survive. Learned topology gain measures $S4 - S4b$ in accuracy points with 95% paired-bootstrap intervals.*

| Model | Search Kept | AUROC | Headroom (pp) | Learned Topology Gain ($S4 - S4b$) [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | ToT@4 | 0.806 | 8.43 | -0.88 [-4.04, 2.21] |
| **Qwen2.5-Math-1.5B** | *None* | 0.922 | 1.88 | +1.11 [0.01, 2.29] |
| **SmolLM2-1.7B** | *None* | 0.529 | 0.60 | +0.06 [-0.29, 0.41] |
| **Qwen2.5-3B** | ToT@4/8 | 0.933 | 3.63 | +1.43 [0.20, 2.84] |
| **Phi-3.5-mini** | ToT@4 | 0.866 | 4.39 | -0.20 [-1.73, 1.33] |
| **Qwen2.5-7B** | ToT@4/8, FoT@8 | 0.784 | 5.99 | **+4.69** [2.09, 7.47] |
| **Pooled, 1.5–1.7B (3)** | — | — | — | +0.10 [-0.98, 1.20] |
| **Pooled, 3–7B (3)** | — | — | — | **+1.98** [0.83, 3.23] |

---

### Table 6: Difficulty Signals and Surface Fit Progression
*Correlation ($|r|$) with single-sample accuracy $\text{Acc}(x, \text{SC}, 1)$ and out-of-fold surface $R^2$.*

| Model | Token $H$ ($r$) | Semantic $H$ ($r$) | Text Features $R^2$ | $+H_{\text{sem}}$ $R^2$ | $+$Probe Features $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-1.5B** | +0.100 | **-0.705** | 0.269 | 0.494 | **0.522** |
| **Qwen2.5-Math-1.5B** | +0.032 | **-0.789** | 0.479 | 0.580 | **0.603** |
| **SmolLM2-1.7B** | -0.349 | **-0.394** | 0.084 | 0.274 | **0.289** |
| **Qwen2.5-3B** | -0.161 | **-0.734** | 0.463 | 0.539 | **0.631** |
| **Phi-3.5-mini** | -0.391 | **-0.639** | 0.210 | 0.265 | **0.465** |
| **Qwen2.5-7B** | -0.305 | **-0.721** | 0.514 | 0.701 | **0.707** |

---

## 4. Figures (Available in `figs/`)

- **[`figs/arch1_pipeline.pdf`](figs/arch1_pipeline.pdf)** (Figure 2 in Manuscript): Architectural overview of the utility-surface allocator pipeline.
- **[`figs/arch2_actionspace.pdf`](figs/arch2_actionspace.pdf)** (Figure 1 in Manuscript): Diagram contrasting 1-D budget selection with 2-D joint topology-budget selection.
- **[`figs/fig3_bootstrap.png`](figs/fig3_bootstrap.png)** (Figure 3 in Manuscript): Paired-bootstrap 95% intervals for allocator gain ($S4b - S0$) and topology gain ($S4 - S4b$).
- **[`figs/fig4_stages_oracles.png`](figs/fig4_stages_oracles.png)** (Figure 4 in Manuscript): Accuracy by stage against both 1-D and 2-D Lagrangian oracles.
- **[`figs/fig_cost_measured.png`](figs/fig_cost_measured.png)** (Figure 5 in Manuscript): Measured token cost of each action against nominal budget $b$.
- **[`figs/fig5_headroom.png`](figs/fig5_headroom.png)** (Figure 6 in Manuscript): Oracle headroom of joint action set over budgets only across targets $B \in \{1.5, 2.0, 3.0\}$.
- **[`figs/fig7_dominance.png`](figs/fig7_dominance.png)** (Figure 7 in Manuscript): Cost-accuracy Pareto frontier and dominance filtering.
- **[`figs/fig6_signals_and_r2.png`](figs/fig6_signals_and_r2.png)** (Figure 8 in Manuscript): Left: absolute correlation of entropy features with accuracy. Right: out-of-fold surface $R^2$ fit progression.

---

## 5. Repository Structure

```text
REASON-X/
├── README.md                          # Complete research report matching publication manuscript
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Python package dependencies
├── environment.yml                    # Conda environment definition
├── .gitignore                         # Excludes checkpoints, caches, and local artifacts
├── notebooks/                         # Clean research experiment notebooks
│   ├── REASON_X_5models.ipynb         # Master experiment notebook (5-solver evaluation)
│   ├── REASON_X_7B_model.ipynb        # Standalone execution for Qwen2.5-7B-Instruct
│   └── REASON_X_incremental_update.ipynb # Multi-model synthesis notebook (3B, 3.8B, 7B)
├── figs/                              # Publication figures
│   ├── src/                           # Figure generation source scripts
│   ├── arch1_pipeline.pdf             # Figure 2: Pipeline diagram
│   ├── arch2_actionspace.pdf          # Figure 1: Action space diagram
│   ├── fig3_bootstrap.png             # Figure 3: Paired bootstrap CIs
│   ├── fig4_stages_oracles.png        # Figure 4: Stage progression vs. oracles
│   ├── fig_cost_measured.png          # Figure 5: Measured cost distributions
│   ├── fig5_headroom.png              # Figure 6: Headroom by budget
│   ├── fig7_dominance.png             # Figure 7: Pareto dominance
│   └── fig6_signals_and_r2.png        # Figure 8: Entropy signals & R2 progression
└── results/                           # Evaluated tabular results & bootstrap outputs
    ├── bootstrap_ci.csv               # 10,000 paired-bootstrap intervals
    ├── costs.csv                      # Per-action measured costs and accuracies
    ├── headline.json                  # Experiment headline metrics and metadata
    ├── ladder.csv                     # Full evaluation ladder across models and budgets
    ├── perq_accuracy.csv              # Per-question accuracy matrix across stages
    ├── table1_budget_only.csv         # Table 2: Budget-only performance
    ├── table2_stages.csv              # Table 3: Accuracy progression by stage
    ├── table3_oracles.csv             # Table 4: Oracle accuracy and headroom
    ├── table4_surviving_actions.csv   # Table 5: Surviving actions and topology gains
    └── table5_entropy_signals.csv     # Table 6: Entropy correlations and R2 fits
```

---

## 6. Data & Checkpoint Availability

- **Hugging Face Hub Repository:**  
  🔗 **[huggingface.co/datasets/shkshreyas/reason-x-ckpt](https://huggingface.co/datasets/shkshreyas/reason-x-ckpt)**
- **Retained Checkpoints:**  
  Utility-tensor checkpoints, tree traces, probe caches, and per-question evaluations for **Qwen2.5-3B-Instruct**, **Phi-3.5-mini-instruct**, and **Qwen2.5-7B-Instruct** are archived on Hugging Face (`ckpt/`).
- **Reproducibility of 1.5–1.7B Solvers:**  
  Checkpoints for the three smaller models (`Qwen2.5-1.5B`, `Qwen2.5-Math-1.5B`, `SmolLM2-1.7B`) were not retained due to storage constraints; they can be regenerated in ~2 hours each on a single T4 GPU using [`notebooks/REASON_X_5models.ipynb`](notebooks/REASON_X_5models.ipynb) with the fixed seed (`CFG.seed = 42`).
- **Tabular Outputs:**  
  All output CSVs, metadata JSON, and pre-computed statistical resamples are directly available in [`results/`](results/).

---

## 7. Replication Instructions

### Zero-Compute Inspection
Open the notebooks in [`notebooks/`](notebooks/) in Jupyter Lab or VS Code to review all pre-computed outputs, tables, and figures:
- **[`notebooks/REASON_X_7B_model.ipynb`](notebooks/REASON_X_7B_model.ipynb)**: Dedicated run for the 7B parameter model.
- **[`notebooks/REASON_X_incremental_update.ipynb`](notebooks/REASON_X_incremental_update.ipynb)**: Multi-model incremental synthesis.
- **[`notebooks/REASON_X_5models.ipynb`](notebooks/REASON_X_5models.ipynb)**: Multi-model evaluation across 1.5B–3.8B solvers.

### Environment Setup
```bash
git clone https://github.com/shkshreyas/REASON-X.git
cd REASON-X

conda env create -f environment.yml
conda activate reason-x
# or: pip install -r requirements.txt

# (Optional) Download cached checkpoints from Hugging Face:
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='shkshreyas/reason-x-ckpt', repo_type='dataset', local_dir='./rx_out/ckpt')"

jupyter lab notebooks/
```

---

## 8. Citation

If you use this codebase, methodology, or computational artifacts, please cite:

```bibtex
@article{kumar2026learning,
  title   = {Learning where to spend test-time compute in language models: a utility-surface allocator and the value of choosing the reasoning topology},
  author  = {Kumar, Shreyas and Rishikeshan, C. A.},
  journal = {Submitted to Knowledge-Based Systems},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/shkshreyas/REASON-X}
}

@software{kumar2026reasonx,
  author    = {Kumar, Shreyas and Rishikeshan, C. A.},
  title     = {{REASON-X: Computational Artifacts and Replication Suite for Utility-Surface Test-Time Compute Allocation}},
  month     = oct,
  year      = 2026,
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.23360219},
  url       = {https://github.com/shkshreyas/REASON-X}
}
```

---

## 9. License

This repository is released under the [MIT License](LICENSE).
