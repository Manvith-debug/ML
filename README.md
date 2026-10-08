# Geothermal score experiments - BT2024261

This project learns two polynomial prediction rules from the BT2024261 datasets.
It uses a single-file command line, a TOML experiment configuration, saved
scikit-learn pipelines, and a short experiment-notebook report.

## Findings

| Dataset | Selected model | Degree | Five-fold CV MSE | Holdout MSE | Holdout R2 |
|---|---|---:|---:|---:|---:|
| var1 | Orthogonal Matching Pursuit (48 terms) | 5 | 0.414685 | 0.292426 | 0.970136 |
| var2 | Ridge with automatic alpha selection | 14 | 0.307495 | 0.250897 | 0.995653 |

Both use ordinary power terms. Chebyshev polynomial terms were also tested,
but produced higher validation errors. The models still meet the requirement
to use polynomial regression: OMP selects polynomial terms and fits their
coefficients, while Ridge fits the full polynomial expansion with a penalty.
The final var2 Ridge alpha is 10; the saved pipeline includes this setting.

## Start here

The existing report, predictions, and saved models are ready to use. No rerun
is needed. To reproduce the experiment later, use Python 3.12 and run:

```bash
python -m pip install -r requirements.txt
python geopoly.py train
python geopoly.py check
python geopoly.py report
```

All project files are directly in this directory. The four personalized CSVs
and `sample_submission.csv` are available locally beside the scripts. Training
and test datasets are excluded from Git by filename. A different data location
can be supplied using `--data PATH`.

To predict again without training:

```bash
python geopoly.py predict --model var1.joblib --input BT2024261_test_var1.csv --output BT2024261_pred_var1.csv
python geopoly.py predict --model var2.joblib --input BT2024261_test_var2.csv --output BT2024261_pred_var2.csv
```

## How the experiment works

**Split once.** Repeated input vectors are kept together. Seed 261 reserves
20% of unique input groups as a holdout. The remaining data use five grouped
validation folds. Var1 has 800 development and 200 holdout rows; var2 has
797 development and 203 holdout rows.

**Try two representations.** Ordinary power terms include squares, cubes,
and interactions. Chebyshev terms use a different polynomial basis: for example,
T2(x) = 2*x^2 - 1. Products are restricted to total degree at most d. With an
intercept, both bases span the same degree-d polynomial space, but penalties
and greedy selection depend on the chosen coordinates.

**Find useful degrees.** The Ridge pipeline tests every degree from 1 to 10
for var1 and from 1 to 20 for var2, with each basis. Features are standardized
inside each fitting fold. RidgeCV chooses alpha from 10 candidates (1e-6 through
1e3) using its efficient leave-one-out criterion inside the fitting partition.
This internal criterion uses the scaler fitted on that entire partition;
the outer validation fold remains untouched. Final alpha can differ from
fold-specific alpha because it is chosen again when fitting all labeled rows.

**Try fewer terms.** At the best Ridge degree for each basis, OMP is tested
with 12, 24, 48, and 96 selected terms. OMP adds a term that explains the current
residual and refits the selected coefficients. This is a prespecified second
stage; it does not exhaustively search OMP at every possible degree.

**Choose and check.** The candidate with the lowest average five-fold MSE
wins. Only then is the reserved holdout evaluated. The selected pipeline is
refitted on all 1,000 training rows and used to predict the 1,000 test rows.
The original data are not changed. The holdout provides an additional check,
while the CV minimum can be optimistic because it was used for selection.

## Project map

```text
geopoly.py              Command-line interface
features.py             Polynomial coordinate transformer
experiment.py           Search, fitting, serialization, inference
checks.py               Numerical and file checks
notebook.py              PDF report generator
settings.toml           Experiment configuration
results.json            Results and source data fingerprints
var1.joblib, var2.joblib Saved fitted pipelines
var*_search.csv         Recorded model comparisons
var*_partitions.csv     Validation split assignments
var*_holdout.csv        Holdout predictions and errors
BT2024261_pred_*.csv    Submission predictions
BT2024261_experiment_notes.pdf
```

`results.json` records settings, scores, software versions, final
alpha, row counts, and dataset SHA-256 fingerprints. Each submission has one
`y` column and preserves the test-row order. Saved pipeline objects contain
the feature transformer, scaler, and regression estimator together.

The check command verifies polynomial terms independently, confirms split
separation, checks the fitted estimators' normal equations, and validates
saved-model predictions, CSV shape, ordering, and recorded metrics.

## Report and repository

The report uses simple explanatory text, serif typography, and vector plots
drawn directly with FPDF2. It includes the repository link by default. Rebuild it
with the following command:

```bash
python geopoly.py report --repository https://github.com/Manvith-debug/ML
```

Repository: [Manvith-debug/ML](https://github.com/Manvith-debug/ML).


The project was flattened without retraining. Existing fitted coefficients,
scalers, prediction CSVs, and recorded scores are preserved. No command needs
to be rerun to use the current deliverables.
