# Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling

Repository for the experiments described in "[Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling](https://www.aclweb.org/anthology/2021.eacl-main.203.pdf)" to be presented at [EACL 2021](https://2021.eacl.org/). Code is released under the MIT license. If you use anything related to the repository or paper, please cite the following work:

```
@inproceedings{emmery-etal-2021-adversarial,
    title = "Adversarial Stylometry in the Wild: 
      {T}ransferable Lexical Substitution Attacks on Author Profiling",
    author = "Emmery, Chris  and
      K{\'a}d{\'a}r, {\'A}kos  and
      Chrupa{\l}a, Grzegorz",
    booktitle = "Proceedings of the 16th Conference of the European Chapter
      of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.203",
    pages = "2388--2402"
}
```

![https://onyx.uvt.nl/sakuin/_public/reap/reap-poster.pdf](https://onyx.uvt.nl/sakuin/_public/reap/reap-poster-prev.png)

## The Paper -- tl;dr

We: 

- ⚔️ Successfully attacked gender classifiers using transformer-based adversarial lexical substitution.
- 🥪 Propose several extensions to [TextFooler](https://github.com/jind11/TextFooler) to propose and rank substitute candidates.
- 🔄 Showed simple substitution models’ attack performances transfer across domains and state-of-the-art models.
- 🚀 Realistic in the wild attack can be collected and fitted locally & fast on a distantly collected corpus.


## Reproduction

Simply run `experiment.py`. That's it; the `__main__` boilerplate has all required code. There's a caveat though (unfortunately): we can't share Twitter data directly. Please contact @cmry if you would like access to the data for reproduction. After, simply insert the password in [this line of code](https://github.com/cmry/reap/blob/main/experiment.py#L243) and everything should\* run.

> \* The code was tested with Python 3.7 on Ubuntu. If anything does not work, please submit an issue.

## Experiment Details

We used a single NVIDIA TITAN X (Pascal) to run the BERT-based models. Loading the data takes approximately half an hour. The normal attacks run in several minutes, the BERT-based models take roughly 2 hours to run on 200 samples given the parameters that we used. This is mainly due to similarity score (which in the current codebase has a non-optimal implementation). With this turned off it runs in less than an hour. For TextFooler, roughly 30G of RAM is required to store the embeddings in memory.

## Dependencies

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
