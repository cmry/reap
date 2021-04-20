from glob import glob
import re
from numpy import mean, std


def get(fn):
    an_lines = []
    for line in open(fn).read().split('\n'):
        if line.startswith('>'):
            an_lines.append(line.split(' ')[2 if 'key' in fn else 1])
    return an_lines


results = {
    'ORG': {'lines': 0, 'correct': [], 'words': []},
    'QTF': {'lines': 0, 'correct': [], 'words': []},
    'QMB': {'lines': 0, 'correct': [], 'words': []},
    'QDB': {'lines': 0, 'correct': [], 'words': []},
    'HTF': {'lines': 0, 'correct': [], 'words': []},
    'HMB': {'lines': 0, 'correct': [], 'words': []},
    'HDB': {'lines': 0, 'correct': [], 'words': []},
}
for v in ['v1', 'v2']:
    v_files = glob(f'./huval/answers/*{v}*')
    key = f'./huval/test-{v}-key.csv'
    for vf in v_files:
        for key_line, a_line in zip(get(key), get(vf)):
            alt, word = tuple(re.sub('[\[\]]', '', a_line).split(','))
            model, words = tuple(re.sub('[\[\]]', '', key_line).split(';'))
            words = set(words.split(','))

            if alt == '1' and model != 'ORG':
                results[model]['correct'].append(1)
            elif alt == '0' and model == 'ORG':
                results[model]['correct'].append(1)
            else:
                results[model]['correct'].append(0)

            if words and word in words:
                results[model]['words'].append(1)
            else:
                results[model]['words'].append(0)

            results[model]['lines'] += 1

for k, v in results.items():
    try:
        out = f"{k}: \t correct: {round(mean(v['correct']), 3)} "
        out += f"({round(std(v['correct']), 3)})\t"
        if not k == 'ORG':
            out += f"\twords: {round(mean(v['words']), 3)} "
            out += f"({round(std(v['words']), 3)})"
        print(out)
    except Exception:
        pass
    fo = open('./graphs/huval.csv', 'w')
    fo.write(f"model{','.join(list(results['ORG'].keys()))}\n")
    for k, v in results.items():
        fo.write(f"{k},{','.join([str(x) for x in v.values()])}\n")
