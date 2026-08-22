# Minimum Structural Repair (MSR)

Reproducible research software for:

**Learning as Minimum Structural Repair: A Variational Principle for Adaptive Systems**

MSR provides a computational realization of a target-relative variational principle: achieve a declared target through the minimum admissible structural change while preserving what already works.

**GitHub Repository:**  
https://github.com/Arithmetic-Power-Geometry/MSR

---

## Overview

Minimum Structural Repair (MSR) asks a different question from conventional error minimization:

> **What is the minimum structural change required to satisfy a target while preserving the existing system as much as possible?**

The software provides a reproducible implementation of this principle for controlled neural adaptation experiments.

The repository includes:

- the MSR Projection algorithm;
- Euclidean-MSR and conventional comparison methods;
- matched neural adaptation experiments;
- repair-geometry sensitivity analysis;
- sequential-repair experiments;
- statistical analyses and bootstrap confidence intervals;
- automatically generated tables and figures;
- unit tests;
- an interactive parameter-exploration application; and
- a one-click GitHub Actions reproduction workflow.

---

## Reproduction on GitHub

Upload the contents of this folder to the root of the GitHub repository.

Then open:

**Actions → Reproduce MSR Results → Run workflow**

The workflow automatically:

1. installs the required dependencies;
2. runs the software tests;
3. reproduces the reported experiments;
4. regenerates the result tables and figures;
5. smoke-tests the interactive application; and
6. uploads the complete `results/` directory as a workflow artifact.

No API key is required.

No external dataset download is required.

The experiments use the **Digits**, **Wine**, and **Breast Cancer Wisconsin Diagnostic** datasets distributed through scikit-learn.

---

## Local Reproduction

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest -q
```

Reproduce all experiments, analyses, tables, and figures:

```bash
python run_all.py
```

---

## Interactive MSR App

Launch the interactive application with:

```bash
python app.py
```

The application allows researchers to vary:

- dataset;
- random seed;
- number of requested edits;
- MSR sensitivity weight `beta`;
- target margin; and
- comparison methods.

For each configuration, the application reports:

- edit success;
- prediction preservation;
- retained accuracy;
- logit drift;
- KL divergence;
- relative parameter change;
- structural repair cost; and
- runtime.

This provides an interactive environment for examining how different repair geometries and adaptation settings affect the trade-off between **target satisfaction** and **preservation**.

---

## Reproduced Outputs

The main generated outputs are stored in `results/`.

### `results/raw_results.csv`

Contains the complete matched benchmark results for all methods, datasets, seeds, and edit conditions.

### `results/paired_tests.csv`

Contains matched statistical comparisons, including:

- paired differences;
- medians;
- bootstrap 95% confidence intervals;
- Wilcoxon tests; and
- win/tie/loss counts.

### `results/beta_sweep.csv`

Evaluates sensitivity to the MSR repair geometry by varying the structural sensitivity parameter `beta`.

### `results/sequential_results.csv`

Contains results from repeated/sequential repair experiments.

### `results/sequential_correlations.csv`

Examines the relationship between cumulative repair burden and collateral change, including comparisons with simpler quantities such as edit count.

### `results/figures/`

Contains the figures generated directly from the reproduced experimental results.

---

## Scientific Scope

Repair Complexity is formulated as a target-relative minimum-repair quantity. MSR Projection is one computational realization of this broader principle for neural adaptation.

The current software provides **controlled proof-of-concept experiments on small neural networks**. These experiments are designed to test the internal predictions of the framework, including target satisfaction, preservation, structural repair cost, geometry sensitivity, and sequential repair.

The present results should not be interpreted as claims of state-of-the-art performance for large-language-model editing, foundation-model adaptation, or every class of adaptive system.

The broader framework is domain-general only when a domain provides meaningful definitions of:

1. the current state;
2. the target or satisfaction condition;
3. the admissible transformations; and
4. the structural cost of change.

---

## Reproducibility

The repository is designed so that the reported computational results can be regenerated from the supplied source code.

The complete reproduction pipeline includes:

**source code → tests → experiments → statistical analysis → tables → figures**

The GitHub Actions workflow provides an independent one-click execution path for this pipeline.

---

## Citation

If you use **Minimum Structural Repair (MSR)**, Repair Complexity, the software, or results from this project, please cite:

**Akhtar, M. A. K. (2026). _Learning as Minimum Structural Repair: A Variational Principle for Adaptive Systems_ (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22058577**

### BibTeX

```bibtex
@misc{akhtar2026msr,
  author    = {Akhtar, M. A. K.},
  title     = {Learning as Minimum Structural Repair: A Variational Principle for Adaptive Systems},
  year      = {2026},
  version   = {V1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22058577},
  url       = {https://doi.org/10.5281/zenodo.22058577}
}
```

---

## Theoretical Foundation

MSR builds on the minimum structural-repair perspective developed in:

**Akhtar, Mohammad Amir Khusru. (2026). _Closure Complexity: A Variational Theory of Minimal Structural Repair_. Available at SSRN: https://ssrn.com/abstract=6734724. https://doi.org/10.2139/ssrn.6734724**

---

## Repository

**Minimum Structural Repair (MSR)**  
https://github.com/Arithmetic-Power-Geometry/MSR

---

## License

Copyright © 2026 Mohammad Amir Khusru Akhtar

Licensed under the **Apache License, Version 2.0**.

The software may be used, reproduced, modified, and distributed subject to the terms of the Apache License 2.0.

See the `LICENSE` file for the complete license text.

---

## Research Principle

> **Achieve the target with the minimum necessary structural change while preserving what already works.**

**One principle. Many domains. Minimum change. Maximum preservation.**
