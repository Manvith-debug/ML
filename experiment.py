"""Configuration-driven comparison with grouped validation and saved pipelines."""
import hashlib
import json
import platform
import tomllib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeCV, OrthogonalMatchingPursuit
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from features import PolynomialCoordinates


def score(y, prediction):
    return {'mse': float(mean_squared_error(y, prediction)), 'r2': float(r2_score(y, prediction))}


def make_pipeline(candidate, settings):
    if candidate['method'] == 'ridge':
        estimator = RidgeCV(alphas=settings['ridge_alphas'], gcv_mode='auto')
    else:
        estimator = OrthogonalMatchingPursuit(n_nonzero_coefs=int(candidate['terms']), precompute=False)
    return Pipeline([('coordinates', PolynomialCoordinates(degree=int(candidate['degree']), basis=candidate['basis'])),
                     ('scale', StandardScaler()), ('regression', estimator)])


def run_project(project, data_dir):
    config = tomllib.loads((project / 'settings.toml').read_text())
    settings = config['experiment']
    outputs = project
    records_dir = project
    project.mkdir(exist_ok=True)
    report = {'roll': settings['roll'], 'seed': settings['seed'], 'validation_folds': settings['validation_folds'],
              'software': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__, 'numpy': np.__version__},
              'problems': {}}
    for name, specification in config['problems'].items():
        path = data_dir / f"{settings['roll']}_train_{name}.csv"
        test_path = data_dir / f"{settings['roll']}_test_{name}.csv"
        labeled, test = pd.read_csv(path), pd.read_csv(test_path)
        columns = [f'x{i+1}' for i in range(specification['inputs'])]
        if labeled.columns.tolist() != columns + ['y'] or test.columns.tolist() != columns:
            raise ValueError(f'Incorrect columns for {name}')
        x, y, xt = labeled[columns].to_numpy(), labeled.y.to_numpy(), test.to_numpy()
        if not all(np.isfinite(a).all() for a in (x, y, xt)):
            raise ValueError('Data must be finite and complete')
        _, groups = np.unique(x, axis=0, return_inverse=True)
        splitter = GroupShuffleSplit(n_splits=1, test_size=settings['holdout_fraction'], random_state=settings['seed'])
        development, holdout = next(splitter.split(x, y, groups))
        fold_splitter = GroupKFold(n_splits=settings['validation_folds'], shuffle=True, random_state=settings['seed'])
        folds = [(development[tr], development[va]) for tr, va in fold_splitter.split(x[development], y[development], groups[development])]
        split_labels = np.full(len(x), 'holdout', dtype=object)
        for number, (_, va) in enumerate(folds):
            split_labels[va] = f'fold_{number+1}'
        pd.DataFrame({'source_row': np.arange(len(x)), 'group': groups, 'partition': split_labels}).to_csv(records_dir / f'{name}_partitions.csv', index=False)
        trials = []
        def evaluate(candidate):
            errors, fitted_alphas = [], []
            for tr, va in folds:
                model = make_pipeline(candidate, settings).fit(x[tr], y[tr])
                errors.append(float(mean_squared_error(y[va], model.predict(x[va]))))
                if candidate['method'] == 'ridge':
                    fitted_alphas.append(float(model[-1].alpha_))
            result = dict(candidate, mean_mse=float(np.mean(errors)), std_mse=float(np.std(errors)),
                          **{f'fold_{j+1}_mse': e for j, e in enumerate(errors)})
            if fitted_alphas:
                result['fold_alphas'] = ','.join(map(str, fitted_alphas))
            trials.append(result)
            pd.DataFrame(trials).to_csv(records_dir / f'{name}_search.csv', index=False)
            print(f"{name}: {candidate['method']:5s} {candidate['basis']:9s} degree={candidate['degree']:2d} terms={candidate.get('terms',0):3d} CV MSE={result['mean_mse']:.6f}", flush=True)
            return result
        print(f'{name}: {len(development)} development / {len(holdout)} holdout rows', flush=True)
        for basis in settings['bases']:
            for degree in range(1, specification['maximum_degree'] + 1):
                evaluate({'method': 'ridge', 'basis': basis, 'degree': degree, 'terms': 0})
        # A second, prespecified stage tests sparse fits at the best dense degree of each basis.
        for basis in settings['bases']:
            dense = min((t for t in trials if t['basis'] == basis and t['method'] == 'ridge'), key=lambda t: t['mean_mse'])
            p = PolynomialCoordinates(dense['degree']).fit(x[:1]).indexer_.n_output_features_
            for terms in sorted(set(min(p, value) for value in settings['sparse_terms'])):
                evaluate({'method': 'omp', 'basis': basis, 'degree': dense['degree'], 'terms': terms})
        selected = min(trials, key=lambda t: (t['mean_mse'], t['degree']))
        diagnostic_model = make_pipeline(selected, settings).fit(x[development], y[development])
        prediction = diagnostic_model.predict(x[holdout])
        pd.DataFrame({'source_row': holdout, 'observed': y[holdout], 'estimated': prediction,
                      'error': y[holdout]-prediction}).to_csv(records_dir / f'{name}_holdout.csv', index=False)
        final = make_pipeline(selected, settings).fit(x, y)
        test_prediction = final.predict(xt)
        if not np.isfinite(test_prediction).all():
            raise ValueError('Nonfinite prediction')
        pd.DataFrame({'y': test_prediction}).to_csv(outputs / f"{settings['roll']}_pred_{name}.csv", index=False)
        joblib.dump({'pipeline': final, 'columns': columns, 'roll': settings['roll']}, project / f'{name}.joblib')
        regression = final[-1]
        information = {'selected': selected, 'holdout': score(y[holdout], prediction),
                       'mean_baseline': score(y[holdout], np.repeat(y[development].mean(), len(holdout))),
                       'development_fit': score(y[development], diagnostic_model.predict(x[development])),
                       'alpha': float(regression.alpha_) if selected['method'] == 'ridge' else None,
                       'polynomial_terms': int(final[0].indexer_.n_output_features_),
                       'nonzero_terms': int(np.count_nonzero(regression.coef_)),
                       'training_rows': len(x), 'test_rows': len(xt), 'development_rows': len(development), 'holdout_rows': len(holdout),
                       'repeated_inputs': int(len(x)-len(np.unique(groups))),
                       'train_boundary_fraction': float((np.abs(x)>=.999999).any(axis=1).mean()),
                       'test_boundary_fraction': float((np.abs(xt)>=.999999).any(axis=1).mean()),
                       'training_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'test_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest()}
        report['problems'][name] = information
        (records_dir / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f"SELECTED {name}: {selected}; holdout={information['holdout']}", flush=True)
    return report


def infer(model_path, input_path, output_path):
    artifact = joblib.load(model_path)
    data = pd.read_csv(input_path)
    if data.columns.tolist() != artifact['columns']:
        raise ValueError('Input columns must match the saved model')
    prediction = artifact['pipeline'].predict(data.to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'y': prediction}).to_csv(output_path, index=False)
