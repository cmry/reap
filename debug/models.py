from ..adversaries.baselines import WeightedRegression, SVM


def get_data():
    X = ['asadads', 'bafssas', 'csaddasdas'] * 300
    y = [0, 1, 0] * 300
    return X, y


def test_wr():
    X, y = get_data()
    wr = WeightedRegression(save=False).fit(X, y)
    assert wr.predict(['asadadsa']) == 0


def test_svm():
    X, y = get_data()
    svm = SVM(tune=True, save=False).fit(X, y)
    assert svm.predict(['asadads']) == 0
