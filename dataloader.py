import re
import csv
import os
import pickle
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib import request, parse

import spacy
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


def authenticate(base, _dir, password):
    """Authenticate for data access."""
    username = 'reap'
    password = password

    if not password:
        exit("Sorry, we can't directly provide tweets for reproduction as " +
             "per Twitter's ToS. Please contact Chris (cmry@pm.me) for the " +
             "password to the data (academic reproduction purposes only).")

    password_mgr = request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, base, username, password)
    handler = request.HTTPBasicAuthHandler(password_mgr)
    r = request.build_opener(handler).open(base + _dir + '.zip')

    if r.status == 200:
        with open(_dir +'.zip', 'wb') as out:
            out.write(r.read())
    else:
        exit("Authentication failed.")


def collect_paper_data(password=None):
    """Collects, unzips, and cleans resources for the experiments."""
    base = 'https://onyx.uvt.nl/sakuin/reap/'
    print("Collecting resources...")
    print(''.join((l.decode('utf-8') for l in
                     request.urlopen(base + 'README.md'))))
    for _dir in ['src', 'data', 'results']:
        print(f"Collecting {_dir} directory...")
        if _dir == 'src':
            request.urlretrieve(base + _dir + '.zip', filename=_dir + '.zip')
        else:
            authenticate(base, _dir, password)
        print(f"Unzipping {_dir}...")
        with zipfile.ZipFile(_dir + '.zip') as z:
            z.extractall(_dir)
        print("Cleaning up...")
        os.remove(_dir + '.zip')


class Preprocessor(object):
    """Textual preprocessing and tokenization using spaCy.
    
    Parameters
    ----------
    spacy_pipe: ``str``, optional (default='en_core_web_sm')
        Changes pipeline for spaCy (see https://spacy.io/models/en).
    """

    def __init__(self, spacy_pipe='en_core_web_sm'):
        self.nlp = spacy.load(spacy_pipe, disable=['parser', 'tagger', 'ner'])

    def _preprocess(self, text):
        """Remove noisy characters and clean twitter-specific tokens."""
        new_text = []
        text = re.sub('[\n]', ' ', text)
        text = re.sub('[ ]+', ' ', text)
        tokens = [token.text.lower() for token in self.nlp(text)]
        for i, token in enumerate(tokens):
            if token == '\t' or token.startswith('http'):
                try:  # peek if sentence did not close correctly
                    if tokens[i - 1] in [';', '.', '?', ':', '!']:
                        token = '\t'
                    else:
                        token = '.'
                except Exception:  # implies this is the starting token
                    token = ''
            if token == '' or token == ' ':  # skip empty tokens
                continue
            if token.startswith('@'):
                token = '__USER__'
            # NOTE: this might introduce some false positives
            elif token == '#':
                token = '__HASHTAG__' + token[1:]
            elif token == '$':
                token = '__STONKS__' + token[1:]
            new_text.append(token)
        new_text = ' '.join(new_text)
        # NOTE: could be done a bit neater (whole function could, honestly)
        new_text = new_text.replace('. .', '.')
        new_text = new_text.replace('.  .', '.')
        return new_text

    def clean(self, text):
        """Map wrapper, removed multiprocessing here due to incompatibility.
        
        Parameters
        ----------
        text: ``list``, required
            List of input texts (str).

        Returns
        -------
        clean_text: ``list``
            List of cleaned up text.
        """
        return list(map(self._preprocess, tqdm(text)))


class Subset(object):
    """Syntactic sugar class."""

    def __init__(self, data, labels):
        self.data = data
        self.target = labels


class Data(object):
    """Syntactic sugar class."""

    def __init__(self, splits):
        X_train, X_test, y_train, y_test = splits
        self.train = Subset(X_train, y_train)
        self.test = Subset(X_test, y_test)


class LabelProcessor(object):
    """Process string labels to binary gender or age multiclass categories.

    Parameters
    ----------
    label: ``str``, required
        String identifier to select class parsing ('gender' or 'age').
    
    Notes
    -----
    We did not run age classifiers for the current set of experiments.
    """

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
        """Transform string label y into integer.
        
        Parameters
        ----------
        y: ``str``, required
            String representation of the label.

        Returns
        -------
        _y, ``str``
            Encoded label representation.
        """
        if self.label == 'gender':
            y = y.lower()[:1]  # can only compare binary gender, alas
            if y == 'm' or y == 'f':
                return y
        elif self.label == 'age':
            if y == 'x':
                return
            return self.age_conversion.get(int(float(y)), 0)
        else:
            return y


class DataLoader(object):
    """CSV data loader that batches text based on user ID.

    Parameters
    ----------
    set_name: ``str``, optional (default=None)
        Name for a pre-made dataset to be loaded. If nothing provided, data
        is assumed to be new and provided by the user.
    delim: ``str``, optional (default=',')
        Delimiter for .csv file that holds the data. CSV assumed to contain
        the following columns: [label, text, user ID].
    label: ``str``, optional (default='gender')
        String representation of the label to be encoded by the
        LabelPreprocessor. Encoding for 'age' and 'gender' is implemented.
    data_dir: ``str``, optional (default='./data')
        Directory where the data can be found, and the pickle files will be
        saved.
    save: ``bool``, optional (default=True)
        If data should be saved as a pickle for faster loading.
    """

    def __init__(self, set_name=None, delim=',', label='gender',
                 data_dir='./data', save=True):
        self.set_name = set_name
        self.delim = delim
        self.label = label
        self.data_dir = data_dir
        self.save = save

    def __str__(self):
        """String representation for printing set names."""
        return str(self.set_name)

    def _batch_user_tweets(self, user_tweets, batch_len=20):
        """Split list of tweets into batches of batch_len."""
        for data in user_tweets:
            label, tweets = data
            tweet_batch = []
            for i, tweet in enumerate(tweets):
                tweet_batch.append(tweet)
                if i and not i % batch_len:
                    yield label, '\t'.join(tweet_batch)
                    tweet_batch = []
            yield label, '\t'.join(tweet_batch)  # join leftover instances

    def _prep_data(self, data, max_len=200):
        """Clean the provided data and cut up to max_len."""
        D_train, D_test, proc, lenc = *data, Preprocessor(), LabelEncoder()

        # NOTE: input is assumed to be tokens
        X_train, y_train = \
            proc.clean(D_train.data), lenc.fit_transform(D_train.target)
        X_test, y_test = proc.clean(D_test.data), lenc.transform(D_test.target)

        X_test, y_test = list(X_test)[-max_len:], list(y_test)[-max_len:]

        return X_train, X_test, y_train, y_test

    def _iter_csv(self, csvf, header_indices, encoder):
        """Loop through csv, encode labels, group per user, and batch."""
        user_tweets = {}
        for row in csvf:
            label, text, uid = tuple(map(lambda i: row[i], header_indices))
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

        for y, batch in self._batch_user_tweets(user_tweets.values()):
            yield y, batch

    def _get_file_info(self):
        """Return file directory and delimiter by set identifier."""
        file_name, delim = None, None
        if self.set_name == 'volk':
            file_name, delim = 'volkova.csv', ','
        elif self.set_name == 'mult':
            file_name, delim = 'corpus.tsv', '\t'
        elif self.set_name == 'query':
            file_name, delim = 'query.csv', ','
        elif self.set_name == 'sklearn':
            file_name = 'sklearn.example'
        else:
            file_name, delim = self.set_name, self.delim
        return file_name, delim

    def _get_data_info(self):
        """Get data and header indices from csv directory."""
        fn, delim = self._get_file_info()
        csvf = csv.reader(open(f'{self.data_dir}/{fn}'), delimiter=delim)
        header = csvf.__next__()
        ix = list(map(lambda x: header.index(x), [self.label, 'text', 'uid']))
        return csvf, ix

    def _load_data(self):
        """Load provided data based on info given in init."""
        lab_proc = LabelProcessor(self.label)
        csvf, ix = self._get_data_info()
        return self._iter_csv(csvf, ix, lab_proc)

    def _load_data_splits(self):
        """Wrapper to load both train and test splits."""
        y, X = zip(*self._load_data())  # NOTE: don't shuffle profiles!
        data = Data(train_test_split(X, y, test_size=0.2,  # stratify=y,
                                     shuffle=False, random_state=42))
        return data.train, data.test

    def _load_sklearn_data(self):
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
        """Load set_name provided data in class init and pickle save.
        
        Returns
        -------
        tweet_batches: ``list``
            List of strings containing batched tweets seperated by a \t.
        """
        save, fn, _, label = False, *self._get_file_info(), self.label
        pickle_dir = f'{self.data_dir}/{fn.split(".")[0] + label + ".pickle"}'

        if self.save:
            if Path(pickle_dir).is_file():
                return pickle.load(open(pickle_dir, 'rb'))
            else:
                save = True

        if self.set_name == 'sklearn':
            data = self._load_sklearn_data()
        else:
            data = self._load_data_splits()

        tweet_batches = self._prep_data(data)
        if save:
            pickle.dump(tweet_batches, open(pickle_dir, 'wb'))

        return tweet_batches

    def corpus_stats(self):
        """Print label, tweet, user, token, and type frequencies for set_name.
        
        Returns
        -------
        corpus_info: ``str``
            String representation of all the corpus descriptives.

        Notes
        -----
        Might wanna run this first time loading data--in all honesty, but hey.
        """
        y, tweets, users, n_tokens, types = Counter(), 0, set(), 0, Counter()
        data_file, header_indices = self._get_data_info()

        for row in data_file:
            label, text, uid = tuple(map(lambda j: row[j], header_indices))
            if label == 'x':
                continue
            tweets += 1
            y[label] += 1
            users.add(uid)
            for token in text.split(' '):
                n_tokens += 1
                types[token] += 1

        X_train, _, _, _ = self.load()
        corpus_info = (
            "Corpus stats\n-----------\n" +
            f"labels: {y}\ntweets: {tweets}\nusers: {len(users)}\n" +
            # NOTE: approximates test set numbers
            f"train: {len(X_train)}\ntest: {(len(X_train)/80)*20}\n" + 
            f"tokens: {n_tokens}\ntypes: {len(types)}\n")
        return corpus_info