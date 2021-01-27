import torch
from sklearn.base import BaseEstimator, TransformerMixin


def pred_probs(model, text, masked=False, is_torch=False, batch_size=32):

    try:
        if type(text) == str:
            text = [text]
        elif type(text[0]) == str:
            text = [' '.join(text)]  # NOTE: test if breaks
        else:
            text = [' '.join(x) for x in text]
    except Exception:
        print("ERROR", text)
        exit()

    if is_torch:
        y_coef = model(text, batch_size=batch_size)
    else:  # sklearn
        pred = model.predict_proba(text)
        y_coef = torch.Tensor(pred)

    if not masked:
        y_pred = torch.argmax(y_coef)
        y_max = y_coef.max()
    else:
        y_pred = torch.argmax(y_coef, dim=-1)
        y_max = y_coef.max(dim=-1)[0]

    return y_coef, y_pred, y_max


class ConcatenateTokens(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(x) for x in X]

    def fit_transform(self, X, y=None):
        return self.transform(X)


class Vocabulary(object):

    def __init__(self):
        self.index2word = {0: '<oov>'}
        self.word2index = {'<oov>': 0}
        self.index = 1

    def fit(self, inputs):
        if isinstance(inputs, str):
            if inputs not in self.word2index:
                if inputs != '':  # NOTE: weird last-word bug
                    self.index2word[self.index] = inputs
                    self.word2index[inputs] = self.index
                    self.index += 1
        if isinstance(inputs, list):
            [self.fit(token) for token in inputs]
        return self

    def transform(self, inputs):
        if isinstance(inputs, str):
            return self.word2index.get(inputs, 0)
        if isinstance(inputs, int):
            return self.index2word.get(inputs, '<oov>')
        if isinstance(inputs, list):
            return [self.transform(token) for token in inputs]

    def fit_transform(self, inputs):
        self.fit(inputs)
        return self.transform(inputs)

    def from_file(self, file_path, embeddings=False):
        with open(file_path, 'r') as document:
            # FIXME: could be tokenized properly
            for line in (x.split(' ') for x in document.read().split('\n')):
                self.fit(line[:1] if embeddings else line)
        return self

    def key_sorted_values(self, to_sort='index'):
        if isinstance(to_sort, str):
            to_sort = self.word2index if to_sort == 'word' else self.index2word
        return [v for k, v in sorted(to_sort.items(), key=lambda x: x[0])]
