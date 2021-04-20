from ..attacks.utils import Vocabulary
from ..attacks.similarities import WordSimilarity, BertSimilarity


def test_load_voc():
    voc = Vocabulary()
    tokens = "this is a text about some stuff".split()

    for word in tokens:
        voc.fit(word)

    return voc


def test_if_fit_works():
    voc = Vocabulary()
    voc.fit("test")
    x = voc.transform("test")
    assert isinstance(x, int)


def test_batch_load():
    voc = Vocabulary()
    tokens = "this is a text about some stuff".split()
    voc.fit(tokens)

    assert len(voc.index2word) == len(tokens) + 1


def test_voc():
    voc = test_load_voc()
    tokens = "this is a text about some stuff".split()

    assert len(voc.index2word) == len(tokens) + 1  # NOTE: oov
    assert len(voc.word2index) == len(tokens) + 1


def test_voc_idx():
    voc = test_load_voc()
    idx = [voc.transform(w) for w in "this is a text about more stuff".split()]
    assert 0 in idx
    assert idx[-1] == len(idx)


def test_batch_idx():
    voc = test_load_voc()
    idx = voc.transform("this is a text about more stuff".split())

    assert 0 in idx
    assert idx[-1] == len(idx)


def test_sorted_voc():
    voc = test_load_voc()
    sorted_voc = voc.key_sorted_values()

    assert sorted_voc[0] == '<oov>'
    assert sorted_voc[-1] == 'stuff'


def test_sim():
    print()
    embedding_file = ('./fitted-embeddings.txt')
    ws = WordSimilarity(embedding_file)

    w1 = ws.voc.transform("blue")
    w2 = ws.voc.transform("red")
    w3 = ws.voc.transform("horse")

    print(w1, w2, w3)
    print(ws.sim[w1, w2], ws.sim[w1, w3])
    assert ws.sim[w1, w2] > ws.sim[w1, w3]


def test_sent_sim():
    print()
    embedding_file = ('./fitted-embeddings.txt')
    ws = WordSimilarity(embedding_file)

    syms = ws.propose_perturbations("the blue and red horse".split())

    assert 'horsey' in syms[-1]


def test_token_similarity():
    print()
    ws = WordSimilarity('./fitted-embeddings.txt')

    print(ws.propose_perturbations(['father', 'mother', 'giraffe']))

    sim1 = ws.token_similarity('father', 'mother')
    sim2 = ws.token_similarity('father', 'stuff')

    assert sim1 < sim2


def replace_sent(bs, tokens, perturb=None, method='dropout', dropout=None):
    if not perturb:
        perturb = tokens
    syms = bs.find_synonyms(perturb, tokens, method=method, top_n=len(tokens),
                            dropout=dropout)
    for i in range(len(tokens)):
        yield syms[i][0] if tokens[i] in perturb else tokens[i]


def test_bert_dropout_similarity():
    print()
    bs = BertSimilarity()

    tokens = "ride the blue and red horse .".split()
    print(' '.join(replace_sent(bs, tokens, dropout=0.5)))

    tokens = "i am not sure how this will turn out .".split()
    print(' '.join(replace_sent(bs, tokens, dropout=0.5)))

    tokens = "i like to cut my beard and drink beer .".split()
    perturb = ["cut", "beard", "beer"]
    print(' '.join(replace_sent(bs, tokens, perturb, dropout=0.5)))


def test_bert_masked_similarity():
    print()
    bs = BertSimilarity()

    tokens = "ride the blue and red horse .".split()
    print(' '.join(replace_sent(bs, tokens, method='masked')))

    tokens = "i am not sure how this will turn out .".split()
    print(' '.join(replace_sent(bs, tokens, method='masked')))

    tokens = "i like to cut my beard and drink beer .".split()
    perturb = ["cut", "beard", "beer"]
    print(' '.join(replace_sent(bs, tokens, perturb, method='masked')))


def test_bert_stuff():
    print()
    from copy import deepcopy as cp

    bs = BertSimilarity()

    tokens = "i like to cut my beard and drink beer .".split()
    perturb = ["cut", "beard", "beer"]
    print(' '.join(tokens))
    print(' '.join(replace_sent(bs, cp(tokens), cp(perturb), method='masked')))
    print(' '.join(replace_sent(bs, cp(tokens), cp(perturb), method='dropout',
                                dropout=0.3)))
    print(' '.join(replace_sent(bs, cp(tokens), cp(perturb), method='dropout',
                                dropout=0.5)))
    print(' '.join(replace_sent(bs, cp(tokens), cp(perturb), method='dropout',
                                dropout=0.9)))
