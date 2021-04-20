import warnings
with warnings.catch_warnings():  # NOTE: filter TF garbage
    warnings.filterwarnings("ignore", category=FutureWarning)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow.compat.v1 as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    import tensorflow_hub as hub
    tf.disable_eager_execution()

    from glob import glob
    import numpy as np
    import nltk
    import torch
    from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
    from bert_score import BERTScorer
    from .utils import Vocabulary


class USESimilarity(object):
    """Calculates sentence similarity based on Universal Sentence Encoder [1].

    Parameters
    ----------
    cache_path: ``str``, required
        Path to where the cache of USE's TF graph should be stored.
    min_sim: ``float``, optional (default=0.7)
        Minimal similarity required to remain in candidate list.

    Notes
    -----
    Discussed in Section 3.3 (Part-of-Speech and Document Encoding).

    References
    ----------
    [1] Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco,
        Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes,
        Steve Yuan, Chris Tar, Brian Strope, and Ray Kurzweil. 2018.
        Universal sentence encoder for English. In Proceedings of the 2018
        Conference on Empirical Methods in Natural Language Processing:
        System Demonstrations, pages 169–174, Brussels, Belgium. Association
        for Computational Linguistics.

    """
    def __init__(self, cache_path, min_sim=0.7):
        """Set USE environment."""
        super(USESimilarity, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/" + \
            "universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self._build_graph()
        self.sess.run([tf.global_variables_initializer(),
                       tf.tables_initializer()])

        self.min_sim = min_sim

    def _build_graph(self):
        """TF graph."""
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)

        self.cosine_similarities = \
            tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = \
            tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)

        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def _sent_sim(self, sents1, sents2):
        """Get sentence similarity."""
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

    def apply_mask(self, texts, A_texts, A_probs, y_pred):
        """Apply 0 mask to candidates that are smaller than min sim.

        Parameters
        ----------
        texts: ``list``, required
            List of (str) input documents (D).
        A_texts: ``A_texts``, required
            List of (str) peturbed documents (D').
        A_probs: ``tensor``, required
            Tensor of omission score probabilites (I_{D_i}).
        y_pred : ``int``, required
            List of (int) predicted classes (y).

        Returns
        -------
        mask: ``array``
            Array of dimensions |D'| * |C_t|.
        """
        sims = self._sent_sim([' '.join(texts)] * len(A_texts),
                              list(map(lambda x: ' '.join(x), A_texts)))[0]
        if len(A_probs.shape) < 2:
            A_probs = A_probs.unsqueeze(0)
        masked = (y_pred != torch.argmax(A_probs, dim=-1)).data.cpu().numpy()
        return masked * (sims >= self.min_sim), sims < self.min_sim


class BERTScore(object):
    """Calculates sentence similarity based on BERTScore [2].

    Notes
    -----
    This scoring is used in the evaluation, discussed in Section 4.5 (Metrics).
    """

    def __init__(self):
        """Set BERT scorer to rescale (min_sim to 0)."""
        self.score = BERTScorer(lang="en", rescale_with_baseline=True)
        self.min_sim = 0

    def sent_sim(self, sents1, sents2):
        """BERTscore sim between sents1 and sents2."""
        return self.score.score(sents1, sents2)

    def apply_mask(self, texts, A_texts, A_probs, y_pred):
        """Can also be used as 0 mask icw min_sim (not in paper).

        Parameters
        ----------
        texts: ``list``, required
            List of (str) input documents (D).
        A_texts: ``A_texts``, required
            List of (str) peturbed documents (D').
        A_probs: ``tensor``, required
            Tensor of omission score probabilites (I_{D_i}).
        y_pred : ``int``, required
            List of (int) predicted classes (y).

        Returns
        -------
        mask: ``array``
            Array of dimensions |D'| * |C_t|.
        """
        sims = self.sent_sim([' '.join(texts)] * len(A_texts),
                             list(map(lambda x: ' '.join(x), A_texts)))
        if len(A_probs.shape) < 2:
            A_probs = A_probs.unsqueeze(0)
        masked = (y_pred != torch.argmax(A_probs, dim=-1)).data.cpu().numpy()
        return masked * (sims >= self.min_sim), sims < self.min_sim


class POSSimilarity(object):
    """Part-of-Speech accuracy ('similarity') masking."""

    def __init__(self):
        """Do nothing."""
        pass

    def get_pos(self, sent, tagset='universal'):
        """Get either universal or default POS tags using NLTK."""
        try:
            if tagset == 'default':
                word_n_pos_list = nltk.pos_tag(sent)
            elif tagset == 'universal':
                word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
            _, pos_list = zip(*word_n_pos_list)
        except IndexError:
            print(sent)
            exit()
        return pos_list

    def pos_filter(self, ori_pos, new_pos_list):
        """Zero out POS tags that don't match original (excluding N/V)."""
        same = [True if ori_pos == new_pos or
                (set([ori_pos, new_pos]) <= set(['NOUN', 'VERB']))
                else False for new_pos in new_pos_list]
        return same

    def apply_mask(self, text, word_index, A_texts, M_probs):
        """Apply pos_filter to sub probabilities, zeroing non-matches.

        Parameters
        ----------
        text: ``str``, required
            String input document (D).
        word_index: ``int``, required
            Target word index in D to compare POS tag of.
        A_texts: ``tensor``, required
            List of (str) peturbed documents (D').
        M_probs : ``int``, required
            Tensor of masked (by USE) omission score probabilites (I_{D_i}).

        Returns
        -------
        mask: ``array``
            Array of dimensions |D'| * |C_t|.
        """
        pos = self.get_pos(text)
        _pos = [self.get_pos(new_text)[word_index] for new_text in A_texts]
        pos_mask = np.array(self.pos_filter(pos[word_index], _pos))
        return M_probs * pos_mask, (1 - pos_mask).astype(float)


class BertSimilarity(object):
    """Bert Similarty for Masked [2] and Dropout [3] Attacks, and re-ranking.

    Parameters
    ----------
    dropout: ``float``, optional (default=None)
        If set to None, no dropout is applied, else it's p of dropping weights.
    top_n: ``int``, optional (default=50)
        Maximum length of target words (T).
    sym_candidates: ``int``, optional (default=20)
        Top k synonyms used after BERT sim ranking.
    bert_rank: ``bool``, optional (default=True)
        Use BERT contextual re-ranking yes/no.
    sim_threshold: ``float`` optional (default=0.8)
        Minimum similarity required for synonym (NOT USED IN PAPER).
    device: ``str``, optional (default='cuda:0')
        Device to put BERT on ('cpu', 'cuda:n').

    Notes
    -----
    These attacks are discussed in Section 3.2 (Masked Substitution, Dropout
    Substitution). The re-ranking is discussed in Section 3.3 (BERT
    Similarity).

    References
    ----------
    [2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.
        BERT: pre-training of deep bidirectional transformers for language
        understanding. InProceedings of the 2019 Con-ference of the North
        American Chapter of theAssociation for Computational Linguistics:
        Hu-man Language Technologies, NAACL-HLT 2019,Minneapolis, MN, USA,
        June 2-7, 2019, Volume1 (Long and Short Papers), pages 4171–4186.
        Association for Computational Linguistics.
    [3] Wangchunshu Zhou, Tao Ge, Ke Xu, Furu Wei, and Ming Zhou. 2019. 
        Bert-based lexical substitu-tion. InProceedings of the 57th Annual
        Meeting of the Association for Computational Linguistics, pages
        3368–3373.
    """

    def __init__(self, dropout=None, top_n=50, sym_candidates=20,
                 bert_rank=True, sim_threshold=0.8, device='cuda:0'):
        """Get transformer bert, func could use updating to new pkg version."""
        self.b = "bert-base-uncased"
        self.d = device
        config = AutoConfig.from_pretrained(self.b, output_hidden_states=True,
                                            output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.b)
        self.model = AutoModelWithLMHead.from_pretrained(self.b, config=config)
        self.model = self.model.to(self.d).eval()
        self.dropout = dropout  # else masked
        self.top_n = top_n
        self.n_sym = sym_candidates
        self.bert_rank = bert_rank
        self.sim_threshold = sim_threshold

    def __str__(self):
        """String representation of BERT sim class for print."""
        return self.__class__.__name__ + \
            ("Masked" if not self.dropout else f"Dropout-{self.dropout}")

    def _dropout_token_embedding(self, sequence, token_idx, p=0.3):
        """Apply dropout to the embedding of the target token."""
        dropout = torch.nn.Dropout(p=p)
        emb = eval(f"self.model.{self.b.split('-')[0]}.embeddings")(sequence)
        emb[:, token_idx] = dropout(emb[:, token_idx])
        return emb

    def _masked_candidates(self, sequence, n_candidates=20, mask=None,
                           dropout=None):
        """Apply mask, get logits, return tokp from logits for n_candidates."""
        if mask is None:
            mask_idx = torch.where(sequence == self.tokenizer.mask_token_id)[1]
        else:
            mask_idx = mask

        if dropout:
            with torch.no_grad():
                drop_embedding = self._dropout_token_embedding(
                    sequence, mask_idx, p=dropout)
                token_logits = self.model(inputs_embeds=drop_embedding)[0]
            logits = token_logits[:, mask_idx, :]
        else:
            with torch.no_grad():
                token_logits = self.model(sequence)[0]
            logits = token_logits[0, mask_idx, :]

        try:
            return torch.topk(logits, n_candidates, dim=1).indices[0].tolist()
        except RuntimeError:
            return []

    def _flat_encode(self, text, mask=False):
        """Syntactic sugar to encode a 1-dim instance."""
        new_text = list(text)
        if isinstance(mask, int):
            new_text[mask] = self.tokenizer.mask_token
        return self.tokenizer.encode(new_text, return_tensors="pt").to(self.d)

    def _bert_sim(self, sentence, cand, ix):
        """Get BERT sim between sent and cand (Eq 2 in paper)."""
        seq = self.tokenizer.encode(sentence, return_tensors="pt").to(self.d)
        sentence[ix] = cand
        _seq = self.tokenizer.encode(sentence, return_tensors="pt").to(self.d)

        logits, pooler, hidden, attention = \
            eval(f"self.model.{self.b.split('-')[0]}")(seq)
        _logits, _pooler, _hidden, _attention = \
            eval(f"self.model.{self.b.split('-')[0]}")(_seq)

        # Equation (2) ----------
        w_ik = torch.mean(torch.stack([layer[:, :, :, ix + 1] for
                          layer in attention]), dim=3)
        score_l = []
        for i in range(len(seq)):
            score_l.append(
                torch.mul(w_ik[i], torch.nn.functional.cosine_similarity(
                    torch.cat(hidden[-5:-1], dim=2)[:, i, :],
                    torch.cat(_hidden[-5:-1], dim=2)[:, i, :])))

        return torch.stack(score_l).sum()
        # -----------------------

    def sentence_encoding(self, sentence):
        """BERT encode sequence."""
        seq = self.tokenizer.encode(sentence, return_tensors="pt").to(self.d)
        logits, pooler, hidden, attention = \
            eval(f"self.model.{self.b.split('-')[0]}")(seq)
        return torch.cat(hidden[-4:], dim=2)

    def _bert_sim_ranking(self, candidates, original_text, ix):
        synonyms = {}
        for token in candidates:
            current_word = self.tokenizer.decode([token])
            synonyms[current_word] = \
                self._bert_sim(original_text, current_word, ix).item()

        # NOTE: can enable > self.sim_threshold after `if x[1]`
        return [x[0] for x in sorted(synonyms.items(), key=lambda x: x[1],
                                     reverse=True) if x[1]]

    def propose_perturbations(self, to_perturb, text):
        """Given text, propose perturbations according to BERT models.

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
        text = text.split(' ')  # FIXME: changed
        to_perturb = list(to_perturb[:])[:self.top_n]

        for idx in range(len(text)):
            if text[idx] in to_perturb:
                if not self.dropout:
                    sequence = self._flat_encode(text, mask=idx)
                    top_candidates = self._masked_candidates(sequence)
                # get embeddings of normal+masked and sim + rank
                else:
                    sequence = self._flat_encode(text)
                    top_candidates = self._masked_candidates(
                        sequence, dropout=self.dropout, mask=idx + 1)

                current_candidate = to_perturb.index(text[idx])
                if self.bert_rank:   # suggests multiple with bert_sim
                    to_perturb[current_candidate] = \
                        self._bert_sim_ranking(top_candidates, text, idx
                                               )[:self.n_sym]
                else:
                    to_perturb[current_candidate] = \
                        [self.tokenizer.decode([x]) for x in top_candidates]

        return to_perturb


class WordSimilarity(object):
    """Textfooler [4] word sim substitution via counter-fitted embeddings [5].

    Parameters
    ----------

    Notes
    -----
    Attack is described in Section 3.2 (Synonym Substitution). Details of
    the embeddings are discussed in Section 4.2.

    If a new vocabulary is given, all OOV words will get a 0-vector.

    References
    ----------
    [4] Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. 2020.
        Is BERT really robust? A strong baseline for natural language attack on
        text classification and entailment. In The Thirty-Fourth AAAI
        Conference on Artificial Intelligence, AAAI2020, The Thirty-Second
        Innovative Applications of Artificial Intelligence Conference,
        IAAI2020, The Tenth AAAI Symposium on Educational Advances in
        Artificial Intelligence, EAAI2020, New York, NY, USA, February 7-12,
        2020, pages 8018–8025. AAAI Press.
    [5] Nikola  Mrkšić, Diarmuid ́O Séaghdha,  Blaise Thomson, Milica
        Gašic, Lina M Rojas Bara-hona, Pei-Hao Su, David Vandyke, Tsung-Hsien
        Wen, and Steve Young. 2016. Counter-fitting word vectors to linguistic
        onstraints. In Proceedings of the 2016 Conference of the North
        American Chapter of the Association for Computational Linguistics:
        Human Language Technologies, pages 142–148.
    """

    def __init__(self, embedding_file='./src/fitted-embeddings.txt',
                 vocabulary=None, save_path='./src', n_synonyms=50,
                 threshold=0.7, to_file=True):
        """Loads similarity matrix."""
        self.save_path = save_path
        self.n_synonyms = n_synonyms
        self.threshold = threshold
        self.voc = vocabulary
        self.sim = self._load_sim_matrix(embedding_file)

    def __str__(self):
        """String representation of class for printing."""
        return self.__class__.__name__

    def _fit_embeddings_to_vocab(self, embedding_file):
        """Extract relevant embeddings for vocabulary."""
        embeddings = {}
        for line in (x.split(' ') for x in open(embedding_file).read(
        ).split('\n')):
            word, value = line[0], [np.float32(x) for x in line[1:]]
            if not value:
                continue
            if not embeddings:  # NOTE: fixes missing <oov> position
                embeddings[0] = [np.float32(0.0000001)] * len(value)
            if not embeddings.get(self.voc.transform(word)):
                embeddings[self.voc.transform(word)] = value

        return self.voc.key_sorted_values(embeddings)

    def token_similarity(self, token1, token2):
        """Get embedding similarity for two tokens."""
        ix1, ix2 = tuple(self.voc.transform([token1, token2]))
        return self.sim[ix1, ix2]

    def _compute_similarities(self, embeddings):
        """Matrix multiplication of emb and emb.T, normalized."""
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return product / np.dot(norm, norm.T)

    def _load_vocab(self, embedding_file):
        """Load vocab from save_path into Vocabulary classes."""
        vocab_file = glob(f'{self.save_path}/voc_list*')

        if vocab_file:
            print("Loading vocab...")
            self.voc = Vocabulary().from_file(vocab_file[0])
        else:
            print("Building vocab...")
            self.voc = Vocabulary().from_file(embedding_file, embeddings=True)

        if self.save_path and not vocab_file:
            with open(f'{self.save_path}/voc_list.txt', 'w') as vl:
                vl.write('\n'.join(self.voc.key_sorted_values()))

    def _load_sim_matrix(self, embedding_file):
        """Load similarity matrix from save_path."""
        matrix_file = glob(f'{self.save_path}/sim_matrix*')

        if not isinstance(self.voc, Vocabulary):  # if no given vocab
            self._load_vocab(embedding_file)

        if matrix_file:
            print("Loading cos sim matrix...")
            sim = np.load(matrix_file[0])
        else:
            print("Building cos sim matrix...")
            embeddings = np.array(
                self._fit_embeddings_to_vocab(embedding_file))
            sim = self._compute_similarities(embeddings)

        if self.save_path and not matrix_file:
            np.save(open(f'{self.save_path}/sim_matrix.pickle', 'wb'), sim)

        return sim

    def propose_perturbations(self, to_perturb, text=None, max_sym=50):
        """Attack target words; can be a single or multipe indices.

        Parameters
        ----------
        to_perturb: ``list``, required
            List of (int) target indices that should be attacked (T).
        text: ``str``, optional (default=None)
            Don't need context, so no original needed. Just syntax complaint.
        max_sym: ``int``, optional (default=50)
            Max amount of synonym candidates (C_t) returned.

        Returns
        -------
        to_perturb: ``list``
            List of list of (str) synonyms C_t per target word T.
        """
        to_perturb = list(to_perturb)
        ix = np.array(self.voc.transform(to_perturb) if
                      isinstance(to_perturb[0], str) else to_perturb)

        sym_index = (np.argsort(-self.sim[ix, :])[:, 1:(1 + max_sym)]).T
        top_syms = (sym_index * (self.sim[[ix], sym_index] > self.threshold)).T
        return [self.voc.transform([int(v) for v in row if v])
                for row in top_syms]
