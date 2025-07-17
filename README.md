# Task complexity shapes internal representations and robustness in neural networks

In this work, we introduce a suite of five data-agnostic probes—pruning, binarization, noise injection, sign flipping, and bipartite network randomization—to quantify how task difficulty influences the topology and robustness of representations in multilayer perceptrons (MLPs). MLPs are represented as signed, weighted bipartite graphs from a network science perspective.

---
---

## Reproducing experiments

- Training neural networks for analysis: `notebooks/train-mnist.ipynb` and `notebooks/train-fashion-mnist.ipynb`. Adjust parameters of the training in `scripts/train_new_subset_fix_acc.py` and `scripts/train_new_subset_fix_acc_fashion_mnist.py`. The SSIM distance is computed in `notebooks/ssim-mnist.ipynb` and `notebooks/ssim-fashion-mnist.ipynb`.

- **Figure 1**: `notebooks/fig1-{DATASET}-prune.ipynb`, `notebooks/fig1-{DATASET}-noise.ipynb`, `notebooks/fig1-{DATASET}-flip.ipynb`
 
- **Figure 2 and 3**: `notebooks/fig2-3-{DATASET}.ipynb`.

- **Figure 4**: `notebooks/fig4a-c-all-digits.ipynb` and `notebooks/fig4b-all-digits.ipynb`.

- **Figure 5**: `notebooks/fig5-distilbert.ipynb`.

where `DATASET = {mnist, fashion-mnist}`.
