# Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling

Repository for the experiments described in "[Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling](https://www.aclweb.org/anthology/2021.eacl-main.203.pdf)" to be presented at [EACL 2021](https://2021.eacl.org/). Code is released under the MIT license. If you use anything related to the repository or paper, please cite the following work:

```bibtex
@inproceedings{emmery-etal-2021-adversarial,
    title = "Adversarial Stylometry in the Wild: {T}ransferable Lexical Substitution Attacks on Author Profiling",
    author = "Emmery, Chris  and  K{\'a}d{\'a}r, {\'A}kos  and  Chrupa{\l}a, Grzegorz",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.203",
    pages = "2388--2402"
}
```

## 📜 Overview

- [🔎 Paper Details](https://github.com/cmry/reap#-paper-details)
    - [🙄 tl;dr](https://github.com/cmry/reap#-tldr)
    - [♻️ Reproduction](https://github.com/cmry/reap#%EF%B8%8F-reproduction)
    - [🚀 Dependencies](https://github.com/cmry/reap#-dependencies)
    - [🛢️ Resources](https://github.com/cmry/reap#%EF%B8%8F-resources)
- [🛠️ Configuration](https://github.com/cmry/reap#%EF%B8%8F-configuration)
- [🔗 Extensions](https://github.com/cmry/reap#-extensionss)
    - [💾 Adding Data](https://github.com/cmry/reap#-adding-data)
    - [⚔️ Adding Attacks](https://github.com/cmry/reap#%EF%B8%8F-adding-attacks)
    - [🥪 Adding Classifiers](https://github.com/cmry/reap#-adding-classifiers)
    - [🔄 Lexical Substitution](https://github.com/cmry/reap#-lexical-substitution)


## 🔎 Paper Details

[![EACL Poster](https://onyx.uvt.nl/sakuin/_public/reap/reap-poster-prev.png)](https://onyx.uvt.nl/sakuin/_public/reap/reap-poster.pdf)

### 🙄 tl;dr

We: 

- ⚔️ Successfully attacked gender classifiers using transformer-based adversarial lexical substitution.
- 🥪 Propose several extensions to [TextFooler](https://github.com/jind11/TextFooler) to propose and rank substitute candidates.
- 🔄 Showed simple substitution models’ attack performances transfer across domains and state-of-the-art models.
- 🚀 Realistic in the wild attack can be collected and fitted locally & fast on a distantly collected corpus.

### ♻️ Reproduction

Simply run `experiment.py`. That's it; the `__main__` boilerplate has all required code. There's a caveat though (unfortunately): we can't share Twitter data directly. Please contact [@cmry](https://github.com/cmry) if you would like access to the data for reproduction. After, simply insert the password in [this line of code](https://github.com/cmry/reap/blob/main/experiment.py#L243) and everything should\* run.

> \* The code was tested with Python 3.7 on Ubuntu. If anything does not work, please submit an issue.

### 🚀 Dependencies

The code was tested using these libraries and versions:

```
bert_score      0.3.6
nltk            3.5
numpy           1.18.4
spacy           2.2.3
scikit-learn    0.23.1       
tensorflow      1.14.0
tensorflow_hub  0.7.0
torch           1.5.0
tqdm            4.46.0
transformers    3.5.1
```

### 🛢️ Resources

We used a single NVIDIA TITAN X (Pascal) to run the BERT-based models. Loading the data takes approximately half an hour. The normal attacks run in several minutes, the BERT-based models take roughly 2 hours to run on 200 samples given the parameters that we used. This is mainly due to similarity score (which in the current codebase has a non-optimal implementation). With this turned off it runs in less than an hour. For TextFooler, roughly 30G of RAM is required to store the embeddings in memory.

## 🛠️ Configuration

The experimental configuration is structured under `main()` in `experiment.py`. Generally, what is described there are a few loops with experimental configurations, for which a static set of substitutions attacks are loaded under the framework of TextFooler. Let's break down what those parts are: 

```python
    # NOTE: this assumes replication of the experiments
    if not Path('./data/query.csv').is_file():
        print("Not all sources were found, let's collect those for you...")
        collect_paper_data(password=None)  # NOTE: for pw see dataloader.py#L2
```

> `experiment.py` -- lines 241-244

This part is to check if the datasets for replication have been put in the `./data` directory. If you do not plan to use these, you can comment this bit out. Next up are the the experiment mode loops that correspond to the [paper](https://aclanthology.org/2021.eacl-main.203.pdf)'s results in Table 3 (also shows info for Table 4):


```python
    # NOTE: runs ALL experiments -- remove loops to narrow down experiments
    for omission_score_model in ['substitute']:  # partial transfer: , 'target'
        for substitute_model_data in ['mult', 'query', 'volk']:  # volk=partial
            for target_model_classifier in ['WeightedRegression', 'NGrAM']:
```

> `experiment.py` -- lines 246-249

There are three levels:
1. The first loop controls the level of transferability on the model side. If ommission scores are generated by the `'target'` model, it's closer to TextFooler (but model-agnostic). For the paper, we tested full transferability (fit omission score model with different data and a different classifier); hence, it's commented out.
2. The second loop controls which datasets the substitute classifier is trained on (the target model is always trained on `'volk'`). Again, for full transferability we want to train on different data, so `'mult'` (Huang et al.) or `'query'` (Emmery et al.) would qualify for that.
3. The third loop determines the architecture the target classifier uses. The substitute classifier is always `'WeightedRegression'`, so `'NGrAM'` will result in full transferability.

Then follows the list of attacks (still Table 3). An excerpt:

```python
perturbators = [
                    TextFooler(HeuristicAttack(mode='nothing'), top1=True),
                    TextFooler(HeuristicAttack(mode='1337'), top1=True),
                    ...
                    TextFooler(BertSimilarity(dropout=0.3)),
                    ...
                    TextFooler(BertSimilarity(dropout=0.3), checks=True)
]
```

> `experiment.py` -- lines 255-272

These attacks can be found under (surprise) `./attacks`. Basically they are objects/classes that all have a `propose_perturbations` method that returns a list of candidates for substitution. In the `perturbators` list they are structured according to the order in Table 3. The `checks` parameter is described in **Section 3.3** of the paper.

Then, finally, we have the last loop to run it all. For every attack we run the above configured experiment:

```python
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
```

> `experiment.py` -- lines 274-284

As you can see, here there are a few default configurations (target classifier) always uses the Volkova et al. set, and the subsitute classifier is always Logistic Regression. The `label` here is important if you want to add your own data (see next section in this README). You can disable `save` and enable `test_only` if you want to do a dry-run on a few instances to check if any new additions are working. How to add these is described below.

## 🔗 Extensions

Our framework can be extended fairly easily using the following steps (all classes can be used in isolation either way, this is if you also want to use the `Experiment` class wrapper from `experiment.py`).

### 💾 Adding Data

The following steps should suffice: 

**Step 1**: add your dataset as a `.csv` in `./data`. The current setup assumes that every file has a column with a name for `label` (set in `Experiment(...)`, last bit of the section above), a column called `text`, and a column with user IDs in `uid`. User IDs are required for batching profiles. You can make these random if you do not want to batch by profile, or you can add your own routine / method by implementing it at this part:
```python
        if self.set_name == 'sklearn':
            data = self._load_sklearn_data()
        else:
            data = self._load_data_splits()
```
> `dataloader.py` -- lines 329-332.

Here, `_load_data_splits` should probably be replaced with something else, and a set name should be added under the conditions. 

**Step 2**: set the `label` and `target_data` parameters of `Experiment(...)` (line 275-... in `experiment.py`) to the name of the column that holds your labels. This should describe well what is in there, as you'll see in the next step. For `target_data` you can give the full file name that is in `./data`.

**Step 3**: if you want to somehow convert the labels, please implement the label name as a condition in the `transform` method of the `LabelProcessor` (starts at line 160 in `dataloader.py`).

That *should* be all.

### ⚔️ Adding Attacks

Attacks are structured as classes. A bare bones example would be:

```python
class ExampleAttack(object):

    def __init__(self): pass
    def __str__(self): return self.__class__.__name__  # for printing / saving

    def propose_perturbations(self, to_perturb, text):
        """Given text, propose perturbations.

        Parameters
        ----------
        to_perturb: ``list``, required
            List of (str) target words that should be attacked (T).
        text: ``str``, required
            Text that should be attacked (D).

        Returns
        -------
        to_perturb: ``list``
            List of list of (str) synonyms C_t per target word T.
        """
        return [[do_something(word)] for word in to_perturb]
```

Note that the example above doesn't use the orginal text. If do want to do so, you should probably add this loop / condition, as the `to_perturb` list are strings rather than indices (not the best implementation I know, mea culpa):

```python
        for idx in range(len(text)):
            if text[idx] in to_perturb:
```

More examples can be found in the `./attacks` directory.

### 🥪 Adding Classifiers

The classifiers are a bit tricky because they use a base model that implements a bunch of standard functionality: tuning, loading and saving models, predicting (labels or class probas). This is implemented under `./targets/baselines.py`. To implement the classifiers themselves, there are two options: using scikit-learn's pipeline (if you do not intend to do tuning), or a slight extension of this if you choose to tune. Let's first focus on the Pipeline example:

```python
class WeightedRegression(Model):
    """Standard tf*idf + LR model."""

    def __init__(self, tune=False, save=True):
        """Standard text classification pipeline."""
        self.clf = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
            ('clf', LogisticRegression(C=1, class_weight='balanced')),
        ])
        ...
        self.save = save
        self.tune = tune
        self.path = "./results/models/{}.pickle"
```

> `./targets/baselines.py` -- lines 93-107 (slightly altered)

Note that the Pipeline object is a class, which is instantiated with a list of tuples. For every tuple, the first element is some string ID you set yourself, the second value is the class of the component. As this is a list, they are run in a sequential fashion (above, TFIDF is applied first, then LR is fitted). The hyperparemeters are set in these class instantiations. Note also that setting `self.clf` with this Pipeline is the only thing you really have to do here, if you have the directory `./results/models` set up (if you want to save).

If you want to tune, the pipeline tuples are now the **keys** of a dictionary, and the parameters are the **values**. The parameters themselves are also dicts, where the key is the string ID you used for the pipeline component, two underscores, and then the name of the hyperparameter you want to tune. Let's suppose we want to tune `C` and `class_weight`, it would look like so:

```python
class WeightedRegressionCV(Model):
    """CV-tuned tf*idf + LR model."""

    def __init__(self, tune=False, save=True):
        """Standard text classification pipeline."""
        self.clf = {
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)): {}
            ('clf', LogisticRegression(): {
                'clf__C': [0.01, 0.1, 1, 10, 100],
                'clf__class_weight': [None, 'balanced']
            }
        }
        ...
        self.save = save
        self.tune = tune
        self.path = "./results/models/{}.pickle"
```

Note that the paremeters are not set in the class initialization anymore, but rather in this `dict` format, with a list of options.

### 🔄 Lexical Substitution

We have a stand-alone version of Zhou et al.'s work in the making. Please stay tuned!