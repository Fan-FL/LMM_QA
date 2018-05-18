# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from numpy import ndarray as nd
import numpy as np
from config import Config
import json
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from string import punctuation
from nltk import word_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag.stanford import StanfordPOSTagger
import math
import codecs
import pickle
from math import log
from gensim.models.keyedvectors import KeyedVectors
from string import punctuation
from nltk import pos_tag, pos_tag_sents


class DataProcessor:
    def __init__(self):
        self.config = Config()
        self.stops = set(stopwords.words("english"))
        self.lmtzr = WordNetLemmatizer()
        self.tagger = StanfordNERTagger(model_filename=self.config.ner_model_path, path_to_jar=self.config.ner_jar_path)
        self.postagger = StanfordPOSTagger(path_to_jar=self.config.pos_jar_path, model_filename=self.config.pos_model_path)
        self.dependency_parser = StanfordDependencyParser(path_to_jar=self.config.parser_jar_path,
                                                          path_to_models_jar=self.config.parser_model_path)

# *********************************** FOR LOAD DATA & MODELS *********************************************


    def load_word2vec_model(self):
        self.model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path, binary=True)

    def load_qtype_model(self):
        with open(self.config.q_type_clf_save_path, 'rb') as f:
            self.qtype_model = pickle.load(f)

    def load_raw_dev(self):
        return self.load_json(self.config.dev_fname)

    def load_raw_test(self):
        return self.load_json(self.config.test_fname)

    def load_raw_train(self):
        return self.load_json(self.config.train_fname)

    def load_raw_doc(self):
        return self.load_json(self.config.doc_fname)

    def load_json(self, fname):
        f = codecs.open(fname, 'rb', encoding='utf-8')
        print fname + ' done'
        data = json.load(f)
        questions = list()
        docids = list()
        answer_paragraphs = list()
        ids = list()
        texts = list()
        for item in data:
            if 'documents' in fname:
                docids.append(item['docid'])
                texts.append(item['text'])
            else:
                questions.append(item['question'])
                docids.append(item['docid'])
                if 'test' in fname:
                    ids.append(item['id'])
                else:
                    texts.append(item['text'])
                    answer_paragraphs.append(item['answer_paragraph'])
        return docids, questions, texts, answer_paragraphs, ids


# *********************************** FOR PREPROCESSING *********************************************


    def lemmatize(self, word):
        word = word.lower()
        lemma = self.lmtzr.lemmatize(word, 'v')
        if lemma == word:
            lemma = self.lmtzr.lemmatize(word, 'n')
        return lemma

    def lemmatize_sent(self, words):
        return [self.lemmatize(word) for word in words]

    def remove_non_alphanumeric(self, words):
        return [re.sub(r'''([^\s\w])+''', '', word) for word in words]

    def ner_sent(self, words):
        ner_sent = self.tagger.tag(words)
        ner_sent = [(tup[0], 'NUMBER') if tup[0].isdigit() else tup for tup in ner_sent]
        ner_sent = [(tup[0], 'OTHER') if tup[1] == 'ORGANIZATION' else tup for tup in ner_sent[:1]] + [(tup[0], 'OTHER') if (tup[0][0].isupper() and tup[1] == 'O') or tup[1] == 'ORGANIZATION' else tup\
                                                                                                    for tup in ner_sent[1:]]
        return ner_sent

    def lower_sent(self, words):
        return [word.lower() for word in words]

    def remove_stop_words(self, words):
        return [word for word in words if word.lower() not in self.stops]

    def remove_mark(self, word):
        return ''.join([c for c in word if c not in punctuation])

    def ner_tagging(self, wiki):
        ner_wiki = self.tagger.tag_sents(wiki)
        new_ners = []
        for i in range(len(ner_wiki)):
            sent = wiki[i]
            pos_sent = pos_tag(sent)
            ner_sent = ner_wiki[i]
            new_ner_sent = []
            for j in range(len(ner_sent)):
                span, tag = ner_sent[j]
                _, pos = pos_sent[j]
                if span.isdigit() or pos == 'CD':
                    new_ner_sent.append((span, 'NUMBER'))
                elif tag == 'ORGANIZATION':
                    new_ner_sent.append((span, 'OTHER'))
                elif j != 0 and tag == 'O' and span[0].isupper():
                    new_ner_sent.append((span, 'OTHER'))
                else:
                    new_ner_sent.append((span, tag))
            new_ners.append(new_ner_sent)
        return new_ners

    def is_all_puncs(self, token):
        if all([x in punctuation for x in token]):
            return True
        return False

    def remove_punc_in_token(self, token):
        return ''.join([x for x in token if x not in punctuation]).strip()

    def preprocess_wiki(self, wiki):
        raw_split = [word_tokenize(sent.replace(u"\u200b",'')) for sent in wiki]
        remove_pure_punc = [[token for token in sent if not self.is_all_puncs(token)] for sent in raw_split]
        remove_punc_in_words = [[self.remove_punc_in_token(token) for token in sent] for sent in remove_pure_punc]
        ner = self.ner_tagging(remove_punc_in_words)
        lower = [self.lower_sent(sent) for sent in remove_punc_in_words]
        remove_stop = [self.remove_stop_words(sent) for sent in lower]
        lemmatized = [self.lemmatize_sent(sent) for sent in remove_stop]
        return remove_pure_punc, ner, lemmatized

    def preprocess_doc_for_training(self, text):
        raw_split = [word_tokenize(par.replace(u"\u200b",'').replace(u"\u2014",'')) for par in text]
        remove_pure_punc = [[token for token in sent if not self.is_all_puncs(token)] for sent in raw_split]
        remove_punc_in_words = [[self.remove_punc_in_token(token) for token in sent] for sent in remove_pure_punc]
        lower = [self.lower_sent(sent) for sent in remove_punc_in_words]
        remove_stop = [self.remove_stop_words(sent) for sent in lower]
        lemmatized = [self.lemmatize_sent(sent) for sent in remove_stop]
        return lemmatized

    def preprocess_questions_for_training(self, question):
        raw_split = word_tokenize(question.replace(u"\u200b", '').replace(u"\u2014", ''))
        remove_pure_punc = [token for token in raw_split if not self.is_all_puncs(token)]
        remove_punc_in_words = [self.remove_punc_in_token(token) for token in remove_pure_punc]
        lower = self.lower_sent(remove_punc_in_words)
        remove_stop = self.remove_stop_words(lower)
        for i in range(len(remove_stop)):
            if remove_stop[i] == 'where':
                remove_stop[i] = 'location'
            if remove_stop[i] == 'who' or remove_stop[i] == 'whom':
                remove_stop[i] = 'person'
            if remove_stop[i] == 'when':
                remove_stop[i] = 'time'
            if remove_stop[i] == 'why':
                remove_stop[i] = 'reason'
        lemmatized = self.lemmatize_sent(remove_stop)
        return lemmatized

    def preprocess_questions(self, raw_qs):
        split = [word_tokenize(sent.replace(u"\u200b",'').replace(u"\u2014",'')) for sent in raw_qs]
        remove_pure_punc = [[token for token in sent if not self.is_all_puncs(token)] for sent in split]
        remove_punc_in_words = [[self.remove_punc_in_token(token) for token in sent] for sent in remove_pure_punc]
        lemmatized = [self.lemmatize_sent(sent) for sent in remove_punc_in_words]
        return lemmatized

    def preprocess_answers(self, answers):
        raw_split = [word_tokenize(sent.replace(u"\u200b", '')) for sent in answers]
        remove_pure_punc = [[token for token in sent if not self.is_all_puncs(token)] for sent in raw_split]
        remove_punc_in_words = [[self.remove_punc_in_token(token) for token in sent] for sent in remove_pure_punc]
        ner = self.ner_tagging(remove_punc_in_words)
        return ner



# *************************************** FOR SENTENCE EXTRACTION MODEL ********************************

    def pair_to_vec(self, sent, q):
        sent_v = nd.tolist(self.sent_to_embedding(sent))
        q_v = nd.tolist(self.sent_to_embedding(q))
        v = sent_v + q_v
        return v

    # return one dimensional array of [par_embedding , q_embedding]
    def pair_to_vec_pre_sent(self, par, q):
        par_v = par
        q_v = nd.tolist(self.par_to_embedding(q))
        v = list(par_v) + list(q_v)
        return v

    def par_to_embedding(self, par):
        if par:
            return np.mean(np.asarray([np.asarray(self.model[word]) if word in self.model.vocab else np.asarray([0.0] * self.config.embedding_size) for word in par]), axis=0)
        else:
            return np.asarray([0.0] * self.config.embedding_size)

    def count_co_occurance(self, par, q):
        count = 0
        for key_word in set(q):
            count += par.count(key_word)
        return 1.0 * count / (len(par) + len(q))

    def n_gram_overlap(self, par, q, n):
        count = 0
        if par < n or q < n:
            return 0
        par_n_grams = []
        q_n_grams = []
        for i in range(len(par) - n + 1):
            n_gram = ' '.join(par[i:i + n])
            par_n_grams.append(n_gram)
        for i in range(len(q) - n + 1):
            n_gram = ' '.join(q[i:i + n])
            q_n_grams.append(n_gram)
        par_n_grams = set(par_n_grams)
        q_n_grams = set(q_n_grams)
        for n_gram in q_n_grams:
            if n_gram in par_n_grams:
                count += 1
        return count

    def longest_co_seq(self, par, q):
        par_raw = ' '.join(par)
        q_raw = ' '.join(q)
        max_size = min([len(par_raw), len(q_raw)])
        for size in range(max_size, 0, -1):
            for i in range(len(par_raw) - size + 1):
                chunk = par_raw[i:i + size]
                if chunk in q_raw:
                    return size
        return 0

    def hand_crafted_features(self, par, q):
        features = []
        features += [self.count_co_occurance(par, q)]
        features += [self.longest_co_seq(par, q)]
        features += [self.n_gram_overlap(par, q, 2)]
        features += [self.n_gram_overlap(par, q, 3)]
        features += [self.n_gram_overlap(par, q, 4)]
        return features

# ********************************* FOR TFIDF ***************************************

    def build_all_invdicts(self, wikis):
        idf_dicts = []
        inv_weighted_dicts = []
        tf_dicts = []
        for wiki in wikis:
            inv_dict, idf_dict, tf_dict = self.build_inverted_indices(wiki)
            idf_dicts.append(idf_dict)
            inv_weighted_dicts.append(inv_dict)
            tf_dicts.append(tf_dict)
        return idf_dicts, inv_weighted_dicts, tf_dicts

    def build_inverted_indices(self, wiki):
        inv_dict = defaultdict(dict)
        idf_dict = defaultdict(int)
        tf_dict = defaultdict(dict)
        N = len(wiki)
        for j in range(N):
            doc = wiki[j]
            for term in doc:
                if not tf_dict[term].has_key(j):
                    tf_dict[term][j] = 0
                tf_dict[term][j] += 1
            for term in set(doc):
                idf_dict[term] += 1
        for term in idf_dict:
            idf = idf_dict[term]
            tf_doc_dict = tf_dict[term]
            inv_dict[term] = {}
            for doc in tf_doc_dict:
                tf = tf_doc_dict[doc]
                tf_idf = tf * math.log(N / idf)
                inv_dict[term][doc] = tf_idf
        return inv_dict, idf_dict, tf_dict

    def rank_wiki_sent_by_query(self, wiki, q, sent_vectors, ranks, rank_dict):
        q_vs = []
        for k in range(len(wiki)):
            if k in ranks:
                rank = ranks.index(k)
            else:
                rank = 100
            sent_v = sent_vectors[k]
            sent = wiki[k]
            pair_v = self.pair_to_vec_pre_sent(sent_v, q)
            val = rank_dict[k]
            pair_v += [val]
            pair_v += [rank]
            pair_v += self.hand_crafted_features(sent, q)
            q_vs.append(pair_v)
        probs = self.clf.predict_proba(q_vs)
        probs_true = [x[1] for x in probs]
        indexs = [i for i in range(len(probs_true))]
        prob_inds = zip(probs_true, indexs)
        ranks = sorted(prob_inds, key=lambda x:x[0], reverse=True)
        ranks = [rank[1] for rank in ranks]
        return ranks


# ********************************* FOR BM25 **********************************************

    # the function return the paragraph(sentence) in document that matches best to the query
    # based on bm25
    # returning a list ranked by descending order
    # like [(3, 21.44802474790178), (18, 3.1944542200601345), (0, 2.4847094773431535),...]
    def get_best_par(self, query, doc):
        inv_dict = defaultdict(dict)
        # the number of paragraphs that contains a given word
        idf_dict = defaultdict(int)
        # term frequency in each paragraph. like{'word1':{1:5,2:4,3:0},'word2':{1:2..}...}
        tf_par_dict = defaultdict(dict)
        # term frequency in query
        tf_query_dict = defaultdict(int)
        acc_bm25_dict = defaultdict(float)

        # total number of paragraphs in this doc
        pars_count = len(doc)
        # average paragraph length in this doc 
        parLength_avg = self.avg_par_length(doc)
        for j in range(pars_count):
            par = doc[j]
            for term in par:
                if j not in tf_par_dict[term].keys():
                    tf_par_dict[term][j] = 0
                tf_par_dict[term][j] += 1
            for term in set(par):
                idf_dict[term] += 1

        for term in query:
            tf_query_dict[term] += 1

        for i in range(pars_count):
            acc_bm25_score = 0.0
            par_Length = len(doc[i])
            for term in query:
                if term in idf_dict.keys():
                    ft = idf_dict[term]
                else:
                    ft = 0
                if term in tf_par_dict.keys():
                    if i in tf_par_dict[term].keys():
                        fdt = tf_par_dict[term][i]
                    else:
                        fdt = 0
                else:
                    fdt = 0
                fqt = tf_query_dict[term]
                acc_bm25_score += self.bm25_score(pars_count, ft, fdt, fqt, par_Length, parLength_avg)
            acc_bm25_dict[i] = acc_bm25_score
        sorted_acc_bm25_dict = sorted(acc_bm25_dict.items(), key=lambda item: item[1], reverse=True)
        return sorted_acc_bm25_dict

    # calculate bm25 scores of a word in a query with a document
    # the bm25 should be accumulated to rank the best document to a query
    # N: number of documents(sentences)
    # ft: number of documents than contain the term
    # fdt: number of a given term in a document
    # fqt: number of a given term in a query
    # Ld: length of the document
    # Ld_avg: average length of the document
    def bm25_score(self, N, ft, fdt, fqt, Ld, Ld_avg, k1=1.2, b=0.75, k3=0):
        idf_component = log((N - ft + 0.5) / (ft + 0.5))
        doc_tf_component = ((k1 + 1) * fdt) / ((k1 - k1 * b + k1 * b * Ld / Ld_avg) + fdt)
        query_tf_component = ((k3 + 1) * fqt) / (k3 + fqt)
        bm25_score = idf_component * doc_tf_component * query_tf_component
        return bm25_score

    def avg_par_length(self, doc):
        N = len(doc)
        totalLen = 0.0
        for par in doc:
            totalLen += len(par)
        return totalLen / N



# ****************** FOR QTYPE MODEL ***************************** #


    def extract_head_word(self, parse):
        for head, _, _ in list(parse.triples()):
            word, tag = head
            if 'NN' in tag and word not in self.stops:
                return word, 0
            if 'VB' in tag and word not in self.stops:
                return word, 1
        return 0, -1

    def dep_parse_sents(self, sents):
        parses_sents = list(self.dependency_parser.parse_sents(sents))
        parses_sents = [list(parses)[0] for parses in parses_sents]
        return parses_sents

    def extract_wh_word(self, words):
        WH_words = self.config.WH_words
        one_hot = [0] * (len(WH_words) + 1)
        for word in words:
            if word in WH_words:
                one_hot[WH_words.index(word)] = 1
                return one_hot
        one_hot[-1] = 1
        return one_hot

    def find_qtype_from_answer(self, answer):
        tag_dict = defaultdict(int)
        for span, tag in answer:
            if tag != 'O':
                tag_dict[tag] += 1
        if tag_dict:
            max_tag = max(tag_dict.items(), key=lambda x:x[1])
            return max_tag[0]
        else:
            return 'O'

    def q_to_vec(self, q, dep_q):
        wh_v = self.extract_wh_word(q)
        q = self.remove_stop_words(q)
        q_v = list(nd.tolist(self.sent_to_embedding(q)))
        head, tag = self.extract_head_word(dep_q)
        if head == 0:
            head_emb = [0] * self.config.embedding_size + [tag]
        else:
            head_emb = list(nd.tolist(self.sent_to_embedding(head))) + [tag]
        return wh_v + q_v + head_emb + [len(q)]

if __name__ == '__main__':
    du = DataProcessor()
