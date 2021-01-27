# Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling

Repository for the experiments described in "Adversarial Stylometry in the Wild: Transferable Lexical Substitution Attacks on Author Profiling" to be presented at EACL 2021. Code is released under the MIT license. If you use anything related to the repository or paper, please cite the following work:

```
TO BE ADDED
```

> **NOTE**: documentation, and general repository information is still being updated. Feel free to check back later for a(n even more) complete version.


## Setup

To reproduce the experiments, collect the required data first, and simply run `experiment.py`. All output can be found under `results` (**to be added**) including the human evaluation and annotation instructions. Instructions to obtain the data can currently be found in `data/README.md` (**bash script will follow**), and for downloading the embeddings in `src/README.md`.


## Experiment Details

We used a single NVIDIA TITAN X (Pascal) to run the BERT-based models. Loading the data takes approximately half an hour. The normal attacks run in several minutes, the BERT-based models take roughly 2 hours to run on 200 samples given the parameters that we used. This is mainly due to similarity score (which in the current codebase has a non-optimal implementation). With this turned off it runs in less than an hour. For TextFooler, roughly 30G of RAM is required to store the embeddings in memory.
