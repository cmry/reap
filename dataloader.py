import re
import csv
import sys
import pickle
from pathlib import Path

import spacy
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


class Preprocessor(object):

    def __init__(self, n_jobs=20):
        self.nlp = spacy.load("en_core_web_sm",
                              disable=['parser', 'tagger', 'ner'])
        self.n_jobs = n_jobs

    def preprocess(self, text):
        new_text = []
        text = re.sub('[\n]', ' ', text)
        text = re.sub('[ ]+', ' ', text)
        tokens = [token.text.lower() for token in self.nlp(text)]
        for i, token in enumerate(tokens):
            if token == '\t' or token.startswith('http'):
                try:
                    if tokens[i - 1] in [';', '.', '?', ':', '!']:
                        token = '\t'
                    else:
                        token = '.'
                except Exception:  # implies this is the starting token
                    token = ''
            if token == '' or token == ' ':
                continue
            if token.startswith('@'):
                token = '__USER__'
            elif re.search('#[A-Za-z0-9]+ ', token):
                token = ' __HASHTAG__ ' + token[1:]
            elif re.search('$[A-Za-z]+ ', token):
                token = ' __STONKS__ ' + token[1:]
            new_text.append(token)
        new_text = ' '.join(new_text)
        new_text = new_text.replace('. .', '.')
        new_text = new_text.replace('.  .', '.')
        return new_text

    def clean(self, text):
        return list(map(self.preprocess, tqdm(text)))


class Subset(object):

    def __init__(self, data, labels):
        self.data = data
        self.target = labels


class Data(object):

    def __init__(self, splits):
        X_train, X_test, y_train, y_test = splits
        self.train = Subset(X_train, y_train)
        self.test = Subset(X_test, y_test)


class LabelProcessor(object):

    def __init__(self, label):
        self.label = label
        categories = {range(13, 16): None,
                      range(16, 20): '2-young zoomer',
                      range(20, 25): '3-old zoomer',
                      range(25, 40): '4-millenial',
                      range(40, 55): '5-genx',
                      range(55, 99): None}
        self.age_conversion = {i: v for k, v in categories.items() for i in k}

    def transform(self, y):
        if self.label == 'gender':
            y = y.lower()[:1]  # can only compare binary gender, alas
            if y == 'm' or y == 'f':
                return y
        elif self.label == 'age':
            if y == 'x':
                return
            return self.age_conversion.get(int(float(y)), 0)


class DataLoader(object):

    def __init__(self, set_name=None, label='gender', save=True):
        self.set_name = set_name
        self.label = label
        self.data_dir = './data'
        self.save = save

    def __str__(self):
        return str(self.set_name)

    def corpus_stats(self):
        from collections import Counter

        y, tweets, users, tokens, types = Counter(), 0, set(), 0, Counter()
        csvf, ix = self.get_data_info()

        for i, row in enumerate(csvf):
            label, text, uid = tuple(map(lambda j: row[j], ix))
            if label == 'x':
                continue
            tweets += 1
            y[label] += 1
            users.add(uid)
            tok = text.split(' ')
            tokens += len(tok)
            for t in tok:
                types[t] += 1

        X_train, X_test, y_train, y_test = self.load()

        print("Corpus stats\n-----------\n" +
              f"labels: {y}\ntweets: {tweets}\nusers: {len(users)}\n" +
              # NOTE: report actual test set numbers instead of aprox
              f"train: {len(X_train)}\ntest: {(len(X_train)/80)*20}\n" + 
              f"tokens: {tokens}\ntypes: {len(types)}\n")

    def batch_user_tweets(self, user_tweets, batch_len=20):
        for user, data in user_tweets.items():
            label, tweets = data
            tweet_batch = []
            for i, tweet in enumerate(tweets):
                tweet_batch.append(tweet)
                if i and not i % batch_len:
                    yield label, '\t'.join(tweet_batch)
                    tweet_batch = []
            yield label, '\t'.join(tweet_batch)  # leftover

    def prep_data_batches(self, data, max_len=200):
        """Clean the provided data and cut up to max_len."""
        D_train, D_test, proc, lenc = *data, Preprocessor(), LabelEncoder()

        # NOTE: input is assumed to be tokens
        X_train, y_train = \
            proc.clean(D_train.data), lenc.fit_transform(D_train.target)
        X_test, y_test = proc.clean(D_test.data), lenc.transform(D_test.target)

        X_test, y_test = list(X_test)[-max_len:], list(y_test)[-max_len:]

        return X_train, X_test, y_train, y_test

    def iter_csv(self, csvf, ix, encoder):
        user_tweets = {}
        for row in csvf:
            label, text, uid = tuple(map(lambda i: row[i], ix))
            try:
                label = encoder.transform(label)
                if not label:
                    continue
            except KeyError:
                exit("Labels incorrectly converted.")
            if not user_tweets.get(uid):
                user_tweets[uid] = (label, [])
            if self.set_name == 'volk':
                for tweet in text.split('\t'):
                    user_tweets[uid][1].append(tweet)
            else:
                user_tweets[uid][1].append(text)

        for y, batch in self.batch_user_tweets(user_tweets):
            yield y, batch

    def get_file_info(self):
        fn, delim = None, None
        if self.set_name == 'volk':
            fn, delim = 'volkova.csv', ','
        elif self.set_name == 'mult':
            fn, delim = 'corpus.tsv', '\t'
        elif self.set_name == 'query':
            fn, delim = 'query.csv', ','
        elif self.set_name == 'sklearn':
            fn = 'sklearn.example'

        return fn, delim

    def get_data_info(self):
        fn, delim = self.get_file_info()
        csvf = csv.reader(open(f'{self.data_dir}/{fn}'), delimiter=delim)
        header = csvf.__next__()
        ix = list(map(lambda x: header.index(x), [self.label, 'text', 'uid']))
        return csvf, ix

    def load_own_set(self):
        lab_proc = LabelProcessor(self.label)
        csvf, ix = self.get_data_info()
        return self.iter_csv(csvf, ix, lab_proc)

    def load_splits(self):
        y, X = zip(*self.load_own_set())  # NOTE: don't shuffle profiles!
        data = Data(train_test_split(X, y, test_size=0.2,  # stratify=y,
                                     shuffle=False, random_state=42))
        return data.train, data.test

    def load_sklearn_data(self):
        """Load binary 20newsgroups data from sklearn."""
        categories = ['sci.crypt', 'sci.space']
        D_train = fetch_20newsgroups(subset='train', categories=categories,
                                     shuffle=True, random_state=42,
                                     remove=('headers', 'footers', 'quotes'))
        D_test = fetch_20newsgroups(subset='test', categories=categories,
                                    shuffle=True, random_state=42,
                                    remove=('headers', 'footers', 'quotes'))
        return D_train, D_test

    def load(self):
        """Wrap own set_name provided data in data classes"""
        save, fn, _, label = False, *self.get_file_info(), self.label
        pickle_dir = f'{self.data_dir}/{fn.split(".")[0] + label + ".pickle"}'

        if self.save:
            if Path(pickle_dir).is_file():
                return pickle.load(open(pickle_dir, 'rb'))
            else:
                save = True

        if self.set_name == 'sklearn':
            data = self.load_sklearn_data()
        else:
            data = self.load_splits()

        batches = self.prep_data_batches(data)
        if save:
            pickle.dump(batches, open(pickle_dir, 'wb'))

        return batches
