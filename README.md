# Minimum Structural Repair (MSR) v2.0

Reproducible software for **Learning as Minimum Structural Repair: A Variational Principle for Adaptive Systems**.

Repository target: https://github.com/Arithmetic-Power-Geometry/MSR

## Reproduction on GitHub

Upload this folder as the repository root. Open **Actions -> Reproduce MSR Results -> Run workflow**. The workflow installs dependencies, runs tests, reproduces every reported result/figure, smoke-tests the interactive app, and uploads the `results/` folder as an artifact.

No API key and no external dataset download are required. Digits, Wine, and Breast Cancer Wisconsin Diagnostic are bundled with scikit-learn.

## Local reproduction

```bash
python -m pip install -r requirements.txt
pytest -q
python run_all.py
```

## Interactive app

```bash
python app.py
```

The app lets a reader change dataset, seed, edit count, MSR sensitivity weight `beta`, target margin, and comparison methods. It reports edit success, prediction preservation, retained accuracy, logit drift, KL drift, relative parameter change, structural cost, and runtime.

## Output files

`results/raw_results.csv` contains the main matched benchmark; `paired_tests.csv` contains matched tests, medians, bootstrap 95% intervals, and win counts; `beta_sweep.csv` tests sensitivity to repair geometry; `sequential_results.csv` tests repeated repair; `sequential_correlations.csv` compares cumulative repair cost with edit count as predictors of collateral change; `figures/` contains manuscript figures.

## Scientific scope

MSR is a proof-of-concept implementation of a target-relative variational principle. The included experiments are controlled small-network tests and do not claim state-of-the-art large-language-model editing performance.

## License

Apache License 2.0.
