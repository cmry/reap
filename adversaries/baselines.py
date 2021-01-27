import warnings
from collections import ChainMap
from pathlib import Path
from hashlib import md5
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    import pickle
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.model_selection import GridSearchCV, KFold


class Model(object):

    def __init__(self, tune=False, save=True):
        self.clf = None
        self.save = save
        self.tune = tune
        self.path = "./results/models/{}.pickle"

    def __str__(self):
        return self.__class__.__name__

    def cv_tune(self, X, y):
        """Big evaluation function, handles oversampling, and cross-val."""
        model = Pipeline(list(self.clf.keys()))
        p_grid = dict(ChainMap(*self.clf.values()))
        grid = GridSearchCV(estimator=model, param_grid=p_grid,
                            cv=KFold(n_splits=2, random_state=42), n_jobs=-1)
        grid.fit(X, y)
        print("\nCV performance:", round(grid.best_score_, 3))

        clf = grid.best_estimator_
        print("\n\nFinal model:\n")
        for step in clf.steps:
            print(step)

        clf.fit(X, y)  # Refit best_estimator_ on the entire train set

        return clf

    def load_model(self):
        self.clf = pickle.load(open(self.path, 'rb'))

    def save_model(self):
        pickle.dump(self.clf, open(self.path, 'wb'))

    def fit(self, X, y):
        hash_id = md5(f"{X[0]}".encode('utf-16')).hexdigest()
        self.path = self.path.format(str(self) + '-' + hash_id)
        if self.save and Path(self.path).is_file():
            return self.load_model()

        if self.tune:
            self.clf = self.cv_tune(X, y)
            model_fit = self.clf
        else:
            model_fit = self.clf.fit(X, y)
        if self.save:
            self.save_model()
        return model_fit

    def predict(self, *args):
        return self.clf.predict(*args)

    def predict_proba(self, *args):
        return self.clf.predict_proba(*args)


class WeightedRegression(Model):

    def __init__(self, tune=False, save=True):
        """Standard text classification pipeline."""
        self.clf = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                      max_df=0.9, use_idf=1, smooth_idf=1,
                                      sublinear_tf=1)),
            ('clf', LogisticRegression(class_weight='balanced',
                                       solver='saga', max_iter=5000)),
        ])
        self.save = save
        self.tune = tune
        self.path = "./results/models/{}.pickle"


class BayesFeatures(BaseEstimator, TransformerMixin):
    """Weights (tf*idf) features with conditional class probabilites.
    Notes
    -----
    Adapted from https://www.kaggle.com/jhoward/.
    """
    def __init__(self) -> None:
        """Set class probabilities."""
        self.r = None

    def pr(self, X, y_i, y):
        """Calculate single class probabilities."""
        p = X[[int(yi == y_i) for yi in y]].sum(0)
        return (p + 1) / (sum([int(yi) == y_i for yi in y]) + 1)

    def fit(self, X, y):
        """Calculate and weight all class probabilities."""
        self.r = np.log(self.pr(X, 1, y) / self.pr(X, 0, y))
        return self

    def transform(self, X):
        """Weight with features with class probabilities."""
        return X.multiply(self.r)

    def fit_transform(self, X, y):
        """Calculate class probabilites and apply as feature weights."""
        self.fit(X, y)
        return self.transform(X)


class NBSVM(Model):

    def __init__(self, tune=True, save=True):
        """Standard text classification pipeline."""
        self.clf = {
            ('vect', TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                     max_df=0.9, use_idf=1, smooth_idf=1,
                                     sublinear_tf=1)): {},
            ('bayes', BayesFeatures()): {},
            ('svc', LogisticRegression(class_weight='balanced',
                                       random_state=42)): {
                        'svc__C': [1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3]
                        }
        }
        self.save = save
        self.tune = tune
        self.path = "./results/models/{}.pickle"


class NGrAM(Model):

    def __init__(self, tune=False, save=True):
        """Standard text classification pipeline."""
        self.clf = Pipeline([
            ('feat', FeatureUnion([
                ('wgram', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                          max_df=1.0, use_idf=1,
                                          sublinear_tf=1)),
                ('cgram', TfidfVectorizer(ngram_range=(3, 5), analyzer='char',
                                          min_df=2, max_df=1.0, use_idf=1,
                                          sublinear_tf=1))])
             ),
            ('svc', LinearSVC(C=1, random_state=42)
             ),
        ])
        self.save = save
        self.tune = tune
        self.path = "./results/models/{}.pickle"
