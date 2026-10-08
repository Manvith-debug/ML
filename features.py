"""Two polynomial coordinate systems behind a scikit-learn transformer."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_array, check_is_fitted


class PolynomialCoordinates(TransformerMixin, BaseEstimator):
    def __init__(self, degree=2, basis='power'):
        self.degree = degree
        self.basis = basis

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.indexer_ = PolynomialFeatures(degree=self.degree, include_bias=False).fit(X)
        if self.basis not in ('power', 'chebyshev'):
            raise ValueError('Unknown polynomial basis')
        return self

    def transform(self, X):
        check_is_fitted(self, 'indexer_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Unexpected input column count')
        if self.basis == 'power':
            return self.indexer_.transform(X)
        # Tensor products T_a(x1) T_b(x2)... where a+b+... <= degree.
        # This spans the same total-degree polynomial space as ordinary powers.
        indices = self.indexer_.powers_
        values = np.ones((len(X), len(indices)))
        for column in range(X.shape[1]):
            values *= chebvander(X[:, column], self.degree)[:, indices[:, column]]
        return values
