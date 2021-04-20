import csv
from glob import glob
from random import choices, seed, shuffle


seed(42)

instructions = """
-------------------------------------------------------------------
INSTRUCTIONS - READ BEFORE YOU START

DISCLAIMER: This data is raw Twitter data, and may contain some profanity and
            possibly other offensive content. We have automatically filtered
            the blatant stuff, but due to it having to be a random sample we
            cannot remove everything. If you are not comfortable with this,
            please indicate you would rather not annotate.

Hi, thanks for participating in our evaluation! Below you will find a set of
tweets---some of them have been altered by an algorithm. You may assume all
sentences were written by speakers of the English language (though of varying
spelling and grammar mastery... it's Twitter after all). We'd like you to
indicate two things between the [] brackets per sentence:

    - does the sentence look, to you, like it was altered by an algorithm?
      (0: no, 1: yes)
    - if so, please note -one- word that made you think this (even though
      there might be multiple), otherwise leave the `-`

If you're unsure, try to fill out something regardless. Note that there are
some special tokens including __ (such as __USER__), these are for
anonymization (@username) and are supposed to be there.

Here are some examples for a quick illustration. Given:

[-,-] trump likes for friendlier european wedding in poland .

Let's assume we think `wedding` looks like it might be changed. We note 1 for
changed, and:

[1,wedding] trump likes for friendlier european wedding in poland .


Another example:

[-,-] a skinned knee is a approve of a childhood well - lived .

`Approve` and `skinned` look out of place and make the sentence quite broken,
but we only note -one word-.

[1,approve] a skinned knee is a approve of a childhood well - lived .


Final example:

[-,-] these decisions will hurt 😭 , but success will soon come 🙌 🏽

Finally! A sentence that looks in order. We note:

[0,-] these decisions will hurt 😭 , but success will soon come 🙌 🏽


Please don't remove the brackets and stick to the formatting as demonstrated
above.

There are 80 sentences to rate; try not to take longer than 10-20 seconds per
sentence (for your own sake). Annotating should take about 20-30 min. Please
rate one sentence at a time and don't scroll back to compare the sentences or
change earlier annotations. We added spacing to avoid accidentally having them
in the same view.

So: algorithmically changed yes (1), or no (0), and if you think the sentence
was changed, note -one word- that's suspicious.
------------------------------------------------------------------



"""

SWEARS = set(open('swears.txt').read().split('\n'))


def plot(targets):
    print(targets)


def select_sentences(rfile):
    rfile.__next__()
    candidates = []
    for i, row in enumerate(rfile):
        ox, ax, oy, ay = tuple(row)
        o_buffer, a_buffer = [], []
        changed = False
        for o_t, a_t in zip(ox.split(' '), ax.split(' ')):
            if o_t == a_t and o_t == '\t':
                o_t, a_t = '.', '.'
            o_buffer.append(o_t)
            a_buffer.append(a_t)
            if o_t != a_t:
                changed = True
            if len(set([x for x in a_buffer if len(x) > 1 and '__' not in x])
                   ) > 10 and changed and o_t == a_t and o_t == '.':
                o_string = ' '.join(o_buffer)
                a_string = ' '.join(a_buffer)
                if '##' not in a_string and 'UNK' not in a_string and \
                   not any([x in SWEARS for x in a_string.split(' ')]):
                    candidates.append((o_string, a_string))
                o_buffer, a_buffer = [], []
            if not changed and o_t == a_t and '.' in a_t:
                o_buffer, a_buffer = [], []
    return list(candidates)


def encode(file_name):
    fid = ''
    if 'query' in file_name:
        fid += 'Q'
    if 'mult' in file_name:
        fid += 'H'
    if 'Word' in file_name:
        fid += 'TF'
    if 'Masked' in file_name:
        fid += 'MB'
    if 'Dropout' in file_name:
        fid += 'DB'
    return fid


def get_samples(targets):
    target_candidates = {}
    for file_name in targets:
        candidates = select_sentences(csv.reader(open(file_name)))
        for i, (original, adversary) in enumerate(candidates):
            if not target_candidates.get(original):
                target_candidates[original] = []
            target_candidates[original].append(
                    (encode(file_name), adversary))
    target_candidates = {('ORG', k): v for k, v in target_candidates.items()
                         if len(v) == len(targets)}
    return target_candidates, choices(list(target_candidates.keys()), k=20)


def piecewise_difference(cand, orig):
    differences = []
    for ct, ot in zip(cand.split(' '), orig.split(' ')):
        if ct != ot:
            differences.append(ct)
    return differences


def split_write(targets, of, gf):
    """
    the annotators write 2 scores: algorithmic change, 1 out-of-place word:
    0: no, 1: yes

    this creates a keyfile where we can easily check each annotation against
    two parts can be validated manually (obfuscation detection parts: model
    and word). we have: Q (query) MB (masked BERT), our key looks like:

    [QMB;set,of,changed,words]
    """

    target_candidates, get = get_samples(targets)
    lines = []
    for k in get:
        org = None
        for model, sent in ([k] + target_candidates[k]):
            edits = []
            if model == 'ORG':
                org = sent
                continue
            else:
                edits = piecewise_difference(sent, org)
            lines.append(f"> [-,-] [{model};{','.join(edits)}] {sent}" +
                         ("\n" * 50))
    shuffle(lines)
    of.write(instructions)
    for line in lines:
        select = line.split(' ')
        of.write(f"{select[0]} {select[1]} {' '.join(select[3:])}")
        gf.write(line)

    originals = []
    for k in get:
        for model, sent in [k]:
            edits = []
            originals.append(f"> [-,-] [{model};{','.join(edits)}] {sent}" +
                             ("\n" * 50))

    for line in originals:
        select = line.split(' ')
        of.write(f"{select[0]} {select[1]} {' '.join(select[3:])}")
        gf.write(line)


def main():
    targets = glob('./gender/mult-*Opt-NoChecks-ally.csv')
    of = open('./huval/test-v1.csv', 'w')
    gf = open('./huval/test-v1-key.csv', 'w')
    split_write(targets, of, gf)

    targets = glob('./gender/query-*Opt-NoChecks-ally.csv')
    of = open('./huval/test-v2.csv', 'w')
    gf = open('./huval/test-v2-key.csv', 'w')
    split_write(targets, of, gf)


if __name__ == "__main__":
    main()
