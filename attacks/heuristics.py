from random import randint


class HeuristicAttack(object):
    """Attacks based on heuristics. Simple stuff.

    Parameters
    ----------
    mode: ``str``, optional (default='space')
        Options: nothing, flip (flips chars), 1337, (leetspeak), space
        (adds spacing).
    scope: ``bool``, optional (default=None)
        This is here for shits and giggles.

    Notes
    -----
    These attacks are discussed in Section 3.2 (Heuristic Substitution).
    """

    def __init__(self, mode='space', scope=None):
        """Generate 1337 dict.
        """
        self.mode = mode
        self.leetspeak = {'i': '1', 'e': '3', 'a': '4', 's': '5', 't': '7',
                          'b': '8', 'g': '9', 'o': '0'}

    def __str__(self):
        """String representation of class for printing."""
        return self.__class__.__name__ + "-" + self.mode

    def transform(self, word, label=None):
        """Transform word based on perturbation modes."""
        if self.mode == 'nothing':
            return word
        if self.mode == 'flip':
            if len(word) < 4:
                return word
            ix = int((len(word) / 2))
            return word.replace(word[ix-1:ix+1], word[ix-1:ix+1][::-1])
        if self.mode == '1337':
            return ''.join([self.leetspeak.get(char, char)
                            for char in list(word)])
        if self.mode == 'space':
            rand_char = randint(0, len(word) - 1)
            char_list = list(word)
            char_list[rand_char] = ' ' + word[rand_char]
            return ''.join(char_list)

    def propose_perturbations(self, to_perturb, text):
        """Attack target words; should be str.

        Parameters
        ----------
        to_perturb: ``list``, required
            List of (str) target words that should be attacked (T).
        text: ``str``, required
            Don't need actual text. Just syntax complaint.

        Returns
        -------
        to_perturb: ``list``
            List of list of (str) synonyms C_t per target word T.
        """
        return [[self.transform(word)] for word in to_perturb]
