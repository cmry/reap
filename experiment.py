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
from targets.baselines import WeightedRegression
from src.scoring import meteor_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Experiment(object):
    """Experiment wrapper to test all transferability configurations.

    Parameters
    ----------
    perturbator: ``class``, required
        Attack class from similarities.py or heuristics.py.
    target_data: ``str``, optional (default='volk')
        Data the target classifier (f) is trained on.
    substitute_data: ``str``, optional (default='query')
        Data the substitute model (f') is trained on.
    target: ``str``, optional (default='standard')
        Architecture (pipeline) the target (f) uses.
    substitute: ``str``, optional (default='standard')
        Architecture (pipeline) the substitute classifier (f') uses.
    label: ``str``, optional (default='gender')
        Column to be used for label.
    mode: ``str``, optional (default='substitute')
        Model to be used to produce omission scores (can be set to target).
    save: ``bool``, optional (default=False)
        Save experiment output to disk.

    Notes
    -----
    Experimental options are:
        target_data -- always trained on volk
        substitute_data -- query or mult (or volk)
        target -- LogisticRegression or NGrAM
        substitute -- always LogisticRegression (could be switched to NGrAM)

    Full tranferability: target NGrAM, substitute_data not volk.
    Full access: target and substitute data/model identical.
    """

    def __init__(self, pertubator, target_data='volk', substitute_data='query',
                 target='standard', substitute='standard', label='gender',
                 mode='substitute', save=False):
        """Load standard experiment, otherwise manual configuration."""

        self.target_data = DataLoader(target_data, label=label).load()
        self.substitute_data = DataLoader(substitute_data, label=label
            ).load() if substitute_data else self.target_data
        self.data_id = substitute_data

        self.mode = mode

        if target == 'standard':
            self.target = WeightedRegression()
        else:
            self.target = target
        if substitute == 'standard':  # NOTE: for explicit decoupling of model
            self.substitute = WeightedRegression()
        elif substitute:
            self.substitute = substitute
        else:
            self.substitute = self.target
        if not pertubator:
            sim = WordSimilarity()
            self.pertubator = TextFooler(sim, is_torch=False)
        else:
            self.pertubator = pertubator

        self.label = label
        self.save = save
        self.bert_score = BERTScore()

    def changes(self, S_data, X_data, avg=True):
        """Check how many changes were made to text."""
        changes = []
        for s_doc, x_doc in zip(S_data, X_data):
            n_changes = 0
            for a, x in zip(s_doc.split(' '), x_doc.split(' ')):
                if a != x:
                    n_changes += 1
            changes.append(n_changes)
        if avg:
            return str(round(np.mean(changes), 0))
        else:
            return changes

    def bertscore(self, S_data, X_data, avg=True):
        """Measure BERTScore."""
        score = self.bert_score.sent_sim(S_data, X_data)[2]
        if avg:
            return round(score.mean().item(), 3)
        else:
            return score

    def meteorscore(self, S_data, X_data, avg=True):
        """Measure METEOR."""
        scores = []
        for s_doc, x_doc in zip(S_data, X_data):
            scores.append(meteor_score([s_doc], x_doc))
        if avg:
            return str(round(np.mean(scores), 3))
        else:
            return scores

    # NOTE: these are probably better in a logger class
    def score(self, target_data, test_data, xid=None):
        """Stringify classification and target classifier accuracy."""
        S_test, s_pred = zip(*target_data)
        X_test, y_test = zip(*test_data)

        if xid:
            writer = csv.writer(open(f'./results/graphs/{xid}.csv', 'w'))
            # NOTE: this could be more neatly integrated (as one call)
            change_list = self.changes(S_test, X_test, avg=False)
            bertsc_list = self.bertscore(S_test, X_test, avg=False)
            meteor_list = self.meteorscore(S_test, X_test, avg=False)
            writer.writerow(['changes', 'bertscore', 'meteor'])
            # ---
            for chng, bert, met in zip(change_list, bertsc_list, meteor_list):
                writer.writerow([chng, bert.item(), met])

        ŷ = self.target.predict(S_test)

        return 'Perturbed target model performance:\n' + \
            "\nAcccuracy:\t" + str(round(accuracy_score(y_test, ŷ), 3)) + \
            "\nBERT F1:\t" + str(self.bertscore(S_test, X_test)) + \
            "\nAvg Changes:\t" + str(self.changes(S_test, X_test)) + \
            "\nMETEOR: \t" + str(self.meteorscore(S_test, X_test)) + "\n\n"

    def progress(self, idx, y_test, text):
        """Print experiment progress."""
        return f"{idx} samples out of {len(y_test)} finished!\n"  # {text}"

    def highlight_edits(self, S_text, text):
        """Highlight the edits made by the pertubator."""
        new_text = ''
        for w1, w2 in zip(text.split(' '), S_text.split(' ')):
            new_text += ('\033[91m' + w2 + '\033[0m ') \
                if w1 != w2 else w1 + ' '
        return f"\n=>\n{new_text}\n"

    def load_exp(self, xid):
        """Load experiment from disk."""
        file_in = open(f'./results/{self.label}/{xid}.csv', 'r')
        reader = csv.reader(file_in, delimiter=',', quotechar='"')
        reader.__next__()  # skip header
        s_x, s_y, o_x, o_y = [], [], [], []
        for row in reader:
            ox, sx, oy, sy = tuple(row)
            oy, ay = int(oy), int(sy)
            o_x.append(ox)
            s_x.append(sx)
            o_y.append(oy)
            s_y.append(ay)

        print(self.score(zip(s_x, s_y), zip(o_x, o_y), xid=xid))
        return zip(s_x, s_y), zip(o_x, o_y)

    def save_exp(self, xid, attacked_output, original_output):
        """Dump experiment and results to file."""
        print(f"Saving output to ./results/{self.label}/{xid}.csv")
        writer = csv.writer(open(f'./results/{self.label}/{xid}.csv', 'w'),
                            delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(['original_text', 'target_text',
                         'original_label', 'target_label'])
        for (sx, sy), (ox, oy) in zip(attacked_output, original_output):
            writer.writerow([ox, sx, oy, sy])

    def run(self, party_reports=False, test_only=False):
        """Wrapper for default experiment runner, can report performances."""
        # NOTE: ay is overwritten
        xid = f"{str(self.data_id)}-{str(self.pertubator)}"
        print(xid)

        for party in ['substitute', 'target']:
            print(f"Fitting {party}...")
            X_train, X_test, y_train, y_test = eval(f'self.{party}_data')
            eval(f'self.{party}.fit')(X_train, y_train)

        # NOTE: last scope state should make X_test, y_test that of target
        if party_reports or 'nothing' in xid:
            for party in ['substitute', 'target']:
                print(f"Original {party} performance...")
                ŷ = eval(f'self.{party}.predict')(X_train)
                print(round(accuracy_score(ŷ, y_train), 3))
                ŷ = eval(f'self.{party}.predict')(X_test)
                print(round(accuracy_score(ŷ, y_test), 3))

        if self.mode == 'substitute':  # FIXME: integrate
            xid += '-substitute'
            omsc = self.substitute
        else:
            xid += '-target'
            omsc = self.target

        if test_only or Path(f'./results/{self.label}/{xid}.csv').is_file():
            return self.load_exp(xid)
        elif test_only:
            exit("Experiment doesn't exist! Set test_only to False.")

        print('Start attacking!')
        attacked_output = []
        # NOTE: didn't test tqdm yet
        for idx, (text, y_true) in tqdm(enumerate(zip(X_test, y_test))):
            # text = text[:500]  NOTE: this is a transformer repo fix
            if not text:
                continue

            S_text, y_pred, S_pred = self.pertubator.attack(omsc, text, y_true)

            # NOTE: DIFFERENT THAN TEXTFOOLER: overwrite substitute prediction
            S_pred = self.target.predict([S_text])[0]
            attacked_output.append((S_text, S_pred))

        print(self.score(attacked_output, zip(X_test, y_test)))
        if self.save:
            self.save_exp(xid, attacked_output, zip(X_test, y_test))


def main():
    """Experiment runner."""
    from attacks.heuristics import HeuristicAttack
    from attacks.similarities import BertSimilarity
    from targets.baselines import WeightedRegression, NGrAM
    from dataloader import collect_paper_data

    # NOTE: this assumes replication of the experiments
    if not Path('./data/query.csv').is_file():
        print("Not all sources were found, let's collect those for you...")
        collect_paper_data(password=None)  # NOTE: for pw see dataloader.py#L26

    # NOTE: runs ALL experiments -- remove loops to narrow down experiments
    for omission_score_model in ['substitute']:  # partial transfer: , 'target'
        for substitute_model_data in ['mult', 'query', 'volk']:  # volk=partial
            for target_model_classifier in ['WeightedRegression', 'NGrAM']:
                print(f"setting: {omission_score_model} -- " +
                      f"{substitute_model_data} -- " +
                      f"{target_model_classifier}")

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
                               substitute_data=substitute_model_data,
                               target_data='volk',
                               substitute=WeightedRegression(),
                               target=target_model_classifier,
                               save=True,
                               label='gender',
                               mode=omission_score_model
                    ).run(test_only=False)
                    perturbators[i] = None  # clear cache


if __name__ == "__main__":
    main()
