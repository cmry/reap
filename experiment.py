import csv
import socket
# reproducibility bit ----------------
from random import seed; seed(42)
from numpy.random import seed as np_seed; np_seed(42)
import os; os.environ['PYTHONHASHSEED'] = str(42)
# -----------------------------------
from pathlib import Path

import numpy as np

from dataloader import DataLoader
from attacks.textfooler import TextFooler
from attacks.similarities import WordSimilarity, BERTScore
from adversaries.baselines import WeightedRegression
from scoring import meteor_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm


try:
    print(open('./logo.md').read() if socket.gethostname() == 'onyx' else '')
except Exception:
    pass


class Experiment(object):
    """Experiment wrapper to test all transferability configurations.

    Parameters
    ----------
    perturbator: ``class``, required
        Attack class from similarities.py or heuristics.py.
    adversary_data: ``str``, optional (default='volk')
        Data the adversary classifier (f) is trained on.
    ally_data: ``str``, optional (default='query')
        Data the substitute model (f') is trained on.
    adversary: ``str``, optional (default='standard')
        Architecture (pipeline) the adversary (f) uses.
    ally: ``str``, optional (default='standard')
        Architecture (pipeline) the substitute classifier (f') uses.
    label: ``str``, optional (default='gender')
        Column to be used for label.
    mode: ``str``, optional (default='ally')
        Model to be used for omission scores (can be set to adversary).
    save: ``bool``, optional (default=False)
        Save experiment output to disk.

    Notes
    -----
    Experimental options are:
        adversary_data -- always trained on volk
        ally_data -- query or mult (or volk)
        adversary -- LogisticRegression or NGrAM
        ally -- always LogisticRegression (could be switched to NGrAM)

    Full tranferability: adversary NGrAM, ally_data not volk.
    Full access: adversary and ally data/model identical.
    """

    def __init__(self, pertubator, adversary_data='volk', ally_data='query',
                 adversary='standard', ally='standard', label='gender',
                 mode='ally', save=False):
        """Load standard experiment, otherwise manual configuration."""

        # FIXME: age has to binned
        self.adversary_data = DataLoader(adversary_data, label=label).load()
        self.ally_data = DataLoader(ally_data, label=label).load() if \
            ally_data else self.adversary_data
        self.data_id = ally_data

        self.mode = mode

        if adversary == 'standard':
            self.adversary = WeightedRegression()
        else:
            self.adversary = adversary
        if ally == 'standard':  # NOTE: for explicit decoupling of the model
            self.ally = WeightedRegression()
        elif ally:
            self.ally = ally
        else:
            self.ally = self.adversary
        if not pertubator:
            sim = WordSimilarity()
            self.pertubator = TextFooler(sim, is_torch=False)
        else:
            self.pertubator = pertubator

        self.label = label
        self.save = save
        self.bert_score = BERTScore()

    def changes(self, A_data, X_data, avg=True):
        """Check how many changes were made to text."""
        changes = []
        for a_doc, x_doc in zip(A_data, X_data):
            n_changes = 0
            for a, x in zip(a_doc.split(' '), x_doc.split(' ')):
                if a != x:
                    n_changes += 1
            changes.append(n_changes)
        if avg:
            return str(round(np.mean(changes), 0))
        else:
            return changes

    def bertscore(self, A_data, X_data, avg=True):
        """Measure BERTScore."""
        score = self.bert_score.sent_sim(A_data, X_data)[2]
        if avg:
            return round(score.mean().item(), 3)
        else:
            return score

    def meteorscore(self, A_data, X_data, avg=True):
        """Measure METEOR."""
        scores = []
        for a_doc, x_doc in zip(A_data, X_data):
            scores.append(meteor_score([a_doc], x_doc))
        if avg:
            return str(round(np.mean(scores), 3))
        else:
            return scores

    # NOTE: these are probably better in a logger class
    def score(self, adversarial_data, test_data, xid=None):
        """Stringify classification and adversarial accuracy."""
        A_test, a_pred = zip(*adversarial_data)
        X_test, y_test = zip(*test_data)

        if xid:
            writer = csv.writer(open(f'./results/graphs/{xid}.csv', 'w'))
            # FIXME: this should be integrated
            change_list = self.changes(A_test, X_test, avg=False)
            bertsc_list = self.bertscore(A_test, X_test, avg=False)
            meteor_list = self.meteorscore(A_test, X_test, avg=False)
            writer.writerow(['changes', 'bertscore', 'meteor'])
            for chng, bert, met in zip(change_list, bertsc_list, meteor_list):
                writer.writerow([chng, bert.item(), met])

        ŷ = self.adversary.predict(A_test)

        return 'Perturbed adversary performance:\n' + \
            "\nAcccuracy:\t" + str(round(accuracy_score(y_test, ŷ), 3)) + \
            "\nBERT F1:\t" + str(self.bertscore(A_test, X_test)) + \
            "\nAvg Changes:\t" + str(self.changes(A_test, X_test)) + \
            "\nMETEOR: \t" + str(self.meteorscore(A_test, X_test)) + "\n\n"

    def progress(self, idx, y_test, text):
        """Print experiment progress."""
        return f"{idx} samples out of {len(y_test)} finished!\n"  # {text}"

    def highlight_edits(self, A_text, text):
        """Highlight the edits made by the pertubator."""
        new_text = ''
        for w1, w2 in zip(text.split(' '), A_text.split(' ')):
            new_text += ('\033[91m' + w2 + '\033[0m ') \
                if w1 != w2 else w1 + ' '
        return f"\n=>\n{new_text}\n"

    def load_exp(self, xid):
        """Load experiment from disk."""
        file_in = open(f'./results/{self.label}/{xid}.csv', 'r')
        reader = csv.reader(file_in, delimiter=',', quotechar='"')
        reader.__next__()  # skip header
        a_x, a_y, o_x, o_y = [], [], [], []
        for row in reader:
            ox, ax, oy, ay = tuple(row)
            oy, ay = int(oy), int(ay)
            o_x.append(ox)
            a_x.append(ax)
            o_y.append(oy)
            a_y.append(ay)

        print(self.score(zip(a_x, a_y), zip(o_x, o_y), xid=xid))
        return zip(a_x, a_y), zip(o_x, o_y)

    def save_exp(self, xid, adversary_output, original_output):
        """Dump experiment and results to file."""
        print(f"Saving output to ./results/{self.label}/{xid}.csv")
        writer = csv.writer(open(f'./results/{self.label}/{xid}.csv', 'w'),
                            delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(['original_text', 'adversary_text',
                         'original_label', 'adversay_label'])
        for (ax, ay), (ox, oy) in zip(adversary_output, original_output):
            writer.writerow([ox, ax, oy, ay])

    def run(self, party_reports=False, test_only=False):
        """Wrapper for default experiment runner, can report performances."""
        # NOTE: ay is overwritten
        xid = f"{str(self.data_id)}-{str(self.pertubator)}"
        print(xid)

        for party in ['ally', 'adversary']:
            print(f"Fitting {party}...")
            X_train, X_test, y_train, y_test = eval(f'self.{party}_data')
            eval(f'self.{party}.fit')(X_train, y_train)

        # NOTE: last scope state should make X_test, y_test that of adversary
        if party_reports or 'nothing' in xid:
            for party in ['ally', 'adversary']:
                print(f"Original {party} performance...")
                ŷ = eval(f'self.{party}.predict')(X_train)
                print(round(accuracy_score(ŷ, y_train), 3))
                ŷ = eval(f'self.{party}.predict')(X_test)
                print(round(accuracy_score(ŷ, y_test), 3))

        if self.mode == 'ally':  # FIXME: integrate
            xid += '-ally'
            adv = self.ally
        else:
            xid += '-adversary'
            adv = self.adversary

        if test_only or Path(f'./results/{self.label}/{xid}.csv').is_file():
            return self.load_exp(xid)
        elif test_only:
            exit("Experiment doesn't exist! Set test_only to False.")

        print('Start attacking!')
        adversary_output = []
        # NOTE: didn't test tqdm yet
        for idx, (text, y_true) in tqdm(enumerate(zip(X_test, y_test))):
            # text = text[:500]  NOTE: this is a transformer repo fix
            if not text:
                continue

            A_text, y_pred, A_pred = self.pertubator.attack(adv, text, y_true)

            # NOTE: DIFFERENT THAN TEXTFOOLER: overwrite ally prediction
            A_pred = self.adversary.predict([A_text])[0]
            adversary_output.append((A_text, A_pred))

        print(self.score(adversary_output, zip(X_test, y_test)))
        if self.save:
            self.save_exp(xid, adversary_output, zip(X_test, y_test))


def main():
    """Experiment runner."""
    from attacks.heuristics import HeuristicAttack
    from attacks.similarities import BertSimilarity
    from adversaries.baselines import WeightedRegression, NGrAM

    # NOTE: runs ALL experiments
    for a_mode in ['ally', 'adversary']:
        # a_mode = 'ally'
        for a_set in ['mult', 'query']:
            for a_clf in ['WeightedRegression', 'NGrAM']:
                print(f"setting: {a_mode} -- {a_set} -- {a_clf}")

                ws = WordSimilarity()
                perturbators = [
                    TextFooler(HeuristicAttack(mode='nothing'), top1=True),
                    TextFooler(HeuristicAttack(mode='1337'), top1=True),
                    TextFooler(HeuristicAttack(mode='flip'), top1=True),
                    TextFooler(HeuristicAttack(mode='space'), top1=True),

                    TextFooler(ws, top1=True),
                    TextFooler(BertSimilarity(), top1=True),
                    TextFooler(BertSimilarity(dropout=0.3), top1=True),

                    TextFooler(ws),
                    TextFooler(BertSimilarity()),
                    TextFooler(BertSimilarity(dropout=0.3)),

                    TextFooler(ws, checks=True),
                    TextFooler(BertSimilarity(), checks=True),
                    TextFooler(BertSimilarity(dropout=0.3), checks=True)
                ]

                for i, perturbator in enumerate(perturbators):
                    Experiment(pertubator=perturbator,
                               ally_data='query', adversary_data='volk',
                               ally=WeightedRegression(),
                               adversary=NGrAM(),
                               save=False, label='gender', mode=a_mode
                               ).run(test_only=False)
                    perturbators[i] = None  # clear cache


if __name__ == "__main__":
    main()
