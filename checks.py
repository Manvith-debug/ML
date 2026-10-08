"""Independent feature, split, optimality, and submission checks."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from features import PolynomialCoordinates


def verify(project, data_dir):
    q = PolynomialCoordinates(degree=2, basis='chebyshev').fit_transform([[.2, -.3]])
    np.testing.assert_allclose(q, [[.2, -.3, -.92, -.06, -.82]], atol=1e-12)
    result = json.loads((project / 'results.json').read_text())
    sample = pd.read_csv(data_dir / 'sample_submission.csv')
    for var, record in result['problems'].items():
        train_path = data_dir / f"{result['roll']}_train_{var}.csv"
        test_path = data_dir / f"{result['roll']}_test_{var}.csv"
        assert hashlib.sha256(train_path.read_bytes()).hexdigest() == record['training_sha256']
        assert hashlib.sha256(test_path.read_bytes()).hexdigest() == record['test_sha256']
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        partitions = pd.read_csv(project / f'{var}_partitions.csv')
        assert partitions.source_row.tolist() == list(range(len(train)))
        assert partitions.groupby('group').partition.nunique().max() == 1
        assert set(partitions.partition) == {'holdout', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5'}
        artifact = joblib.load(project / f'{var}.joblib')
        model = artifact['pipeline']
        assert artifact['columns'] == test.columns.tolist()
        prediction = model.predict(test.to_numpy())
        submitted = pd.read_csv(project / f"{result['roll']}_pred_{var}.csv")
        assert submitted.columns.tolist() == sample.columns.tolist() == ['y']
        assert len(submitted) == len(test) == len(sample)
        assert np.isfinite(submitted.y).all()
        np.testing.assert_allclose(prediction, submitted.y, atol=1e-12, rtol=1e-12)
        transformed = model[:-1].transform(train[artifact['columns']].to_numpy())
        estimator = model[-1]
        residual = train.y.to_numpy() - model.predict(train[artifact['columns']].to_numpy())
        assert abs(residual.mean()) < 1e-8
        # Normal equations independently check each final fitted estimator.
        if record['selected']['method'] == 'omp':
            active = estimator.coef_ != 0
            assert active.sum() == record['selected']['terms']
            np.testing.assert_allclose(transformed[:, active].T @ residual, 0, atol=1e-7)
        else:
            np.testing.assert_allclose(transformed.T @ residual, estimator.alpha_ * estimator.coef_, atol=1e-5, rtol=1e-5)
        holdout = pd.read_csv(project / f'{var}_holdout.csv')
        np.testing.assert_allclose(holdout.observed, train.y.iloc[holdout.source_row])
        np.testing.assert_allclose(np.mean(holdout.error**2), record['holdout']['mse'])
        search = pd.read_csv(project / f'{var}_search.csv')
        np.testing.assert_allclose(search.mean_mse.min(), record['selected']['mean_mse'])
        print(f'PASS {var}: data fingerprints, grouped splits, optimality, saved model, metrics, and submission format.')
    print('PASS: Chebyshev terms match independently calculated values.')
