from .similarities import USESimilarity, POSSimilarity
import torch
import numpy as np
from .utils import pred_probs


class TextFooler(object):
    """TextFooler [1] main attack class (implements BERT attacks).

    Parameters
    ----------
    perturbator: ``class``, required
        Attack class from similarities.py or heuristics.py.
    save_path: ``str``, optional (default='.')
        Doesn't do anything currenty, can be prepended for user-friendliness.
    top1: ``bool``, optional (default=False)
        Return top-1 synonyms only.
    import_score_threshold: ``bool``, optional (default=-1.)
        If set to a value, some minimum omission score is required.
    batch_size: ``int``, optional (default=32)
        Size of the batches. :)
    is_torch: ``bool``, optional (default=False)
        Model that generates omission scores is a torch model (yes/no).
    checks: ``bool``, optional (default=False)
        Apply TextFooler checks (and stopping criterion).

    Notes
    -----
    Most of the experimental conditions are described in Section 4.

    References
    ----------
        [4] Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. 2020.
        Is BERT really robust? A strong baseline for natural language attack on
        text classification and entailment. In The Thirty-Fourth AAAI
        Conference on Artificial Intelligence, AAAI2020, The Thirty-Second
        Innovative Applications of Artificial Intelligence Conference,
        IAAI2020, The Tenth AAAI Symposium on Educational Advances in
        Artificial Intelligence, EAAI2020, New York, NY, USA, February 7-12,
    """

    def __init__(self, perturbator, save_path='.', top1=False,
                 import_score_threshold=-1., sim_score_threshold=0.8,
                 batch_size=32, is_torch=False, checks=False):
        self.perturbator = perturbator
        self.checks = checks
        self.top1 = top1

        self.stop_words = set(open('./src/stop_words.txt').read().split())
        if checks:
            self.USE = USESimilarity('./src/cache',
                                     min_sim=sim_score_threshold)
            self.POS = POSSimilarity()

        self.batch_size = batch_size
        self.min_sim = sim_score_threshold
        self.score_max = import_score_threshold
        self.is_torch = is_torch
        self.n_queries = 0
        print(f"Sim: {sim_score_threshold}")

    def __str__(self):
        """String representation of class for printing."""
        return self.__class__.__name__ + '-' + str(self.perturbator) + \
            ('-Top1' if self.top1 else '-Opt') + \
            ('-Checks' if self.checks else '-NoChecks')

    def _rank_candidates(self, text, y_coef, y_pred, y_max):
        """Ranks words according to importance, selecting which to perturb."""
        text = text.split(' ')
        masked_text = [text[:i] + ['<oov>'] + text[min(i + 1, len(text)):]
                       for i in range(len(text))]
        masked_prob, masked_argmax, masked_max = \
            pred_probs(self.model, masked_text, masked=True,
                       is_torch=self.is_torch, batch_size=self.batch_size)

        self.n_queries += len(masked_text)

        scores = (y_max - masked_prob[:, y_pred] +
                  (masked_argmax != y_pred).float() *
                  (masked_max -
                   torch.index_select(y_coef.squeeze(), 0, masked_argmax))
                  ).data.cpu().numpy()

        return [(ix, text[ix]) for ix, score in
                sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                if score > self.score_max and text[ix] not in self.stop_words]

    def _replace(self, text, sym_candidates, y_pred, y_max):
        """Replace words in text with sym candidates towards label change."""
        text = text.split(' ')
        X, _X = text[:], text[:]

        # NOTE: partly covered in Algorithm 1
        for word_index, synonyms in sym_candidates:
            if not synonyms:
                continue
            if self.top1:
                topc = synonyms[0] if 'UNK' not in synonyms[0] else synonyms[1]
                X[word_index] = topc
                continue

            A_texts = [X[:word_index] + [synonym] +
                       X[min(word_index + 1, len(text)):]
                       for synonym in synonyms]

            A_probs, _, _ = pred_probs(self.model, A_texts)

            self.n_queries += len(A_texts)

            if self.checks:
                M_probs, aux_sim = \
                    self.USE.apply_mask(_X, A_texts, A_probs, y_pred)
                M_probs, aux_pos = \
                    self.POS.apply_mask(text, word_index, A_texts, M_probs)
                if np.sum(M_probs) > 0:
                    X[word_index] = synonyms[(M_probs * aux_sim).argmax()]
                    break
                else:
                    new_A_prob = A_probs[:, y_pred] + torch.from_numpy(
                        aux_sim + aux_pos).float()
            else:
                new_A_prob = A_probs[:, y_pred]

            new_A_prob_min, new_A_prob_argmin = torch.min(new_A_prob, dim=-1)

            # only change if decrease in probability
            if new_A_prob_min < y_max:
                X[word_index] = synonyms[new_A_prob_argmin]

            _X = X[:]

        return X

    def attack(self, model, text, y_true):
        """Attack text using omission score model.

        Parameters
        ----------
        model: ``class``, required
            Classifier (f') that is used for omission scores.
        text: ``str``, required
            Document (D) to attack.
        y_true: ``int``, required
            Original class label.

        Returns
        -------
        X_adv: str
            Adversarial sample (synonym substituted text).
        """
        # first check the prediction of the original text
        self.model = model
        y_coef, y_pred, y_max = pred_probs(model, text, is_torch=self.is_torch,
                                           batch_size=self.batch_size)
        if y_true != y_pred:  # NOTE: if initial label incorrect
            return text, y_pred, y_pred
        idx, to_perturb = \
            zip(*self._rank_candidates(text, y_coef, y_pred, y_max))
        sym_candidates = \
            zip(idx, self.perturbator._propose_perturbations(to_perturb, text))

        X = self._replace(text, sym_candidates, y_pred, y_max)
        final_probs, final_argmax, _ = \
            pred_probs(model, [X], is_torch=self.is_torch,
                       batch_size=self.batch_size)
        return ' '.join(X), y_pred, final_argmax
