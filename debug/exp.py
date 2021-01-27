from ..experiment import Experiment
from ..attacks.heuristics import HeuristicAttack, Heuristic, Pertubator


def test_default_experiment():
    Experiment(adversary=None).run()


def test_volkova_clf_experiment():
    print()
    exp = Experiment(adversary=None, data='volk')
    X_train, X_test, y_train, y_test = exp.prep_data_batches(exp.data)

    from ..attack import ConcatenateTokens
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    cat = ConcatenateTokens()
    cv = CountVectorizer()
    lr = LogisticRegression()

    print(X_train[0])
    Xf = cv.fit_transform(cat.fit_transform(X_train))
    lr.fit(Xf, y_train)

    ŷ = lr.predict(cv.transform(cat.transform(X_test)))
    print(classification_report(ŷ, y_test))


def test_experiment():
    print()
    Experiment(pertubator=None, adversary_data='sklearn',
               ally='standard', adversary='standard'
               ).run()


def test_volkova_standard_experiment():
    print()
    Experiment(pertubator=None, adversary_data='volk', adversary='standard'
               ).run()


def test_cross_experiment():
    print()
    Experiment(pertubator=None, ally_data='mult', adversary_data='volk',
               ally='standard', adversary='standard'
               ).run()


def test_heuristic_baseline():
    Experiment(pertubator=HeuristicAttack(Heuristic),
               ally='standard', adversary_data='sklearn', adversary='standard'
               ).run(fit=True)


def test_heuristic_fast_attack():
    Experiment(pertubator=HeuristicAttack(Pertubator, mode='1337'),
               ally='standard', adversary_data='sklearn', adversary='standard'
               ).run(fit=True)


def test_heuristic_attack():
    Experiment(pertubator=HeuristicAttack(Pertubator, mode='1337'),
               ally_data='mult', adversary_data='volk',
               ally='standard', adversary='standard'
               ).run(fit=True)
