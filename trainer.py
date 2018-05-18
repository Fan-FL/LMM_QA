from config import Config
from data_processor import DataProcessor
import sklearn
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
from nltk.corpus import brown, movie_reviews, treebank
import numpy as np
from numpy import ndarray as nd
import sys
import pickle
from collections import defaultdict

class Trainer:
    def __init__(self, mode=None, load=0):
        self.dataProcessor = DataProcessor()
        self.config = Config()
        self.dataProcessor.load_word2vec_model()
       # self.load_raw_dev()
        self.load_raw_training()
        self.load_raw_doc()
        # self.load_raw_dev()
        # self.load_raw_test()
        # self.process_test_data()
        if load:
            with open('processed_data.pkl', 'rb') as f:
                self.processed_qs, self.processed_doc = pickle.load(f)
        else:
            self.process_data()
        # self.process_dev_data()
        # one dimensional vector of [paragraph_embedding , query_embedding, bm25_score, hand_crafted_features]
        self.training_vectors = []
        # 1 for correct paragraph and 0 for wrong paragraph
        self.labels = []
        self.build_training_data()
        self.train_answer_paragraph_selection()
        # self.evaluate()


    def load_raw_training(self):
        self.docids, self.questions, self.answers, self.answer_paragraphs, _ = self.dataProcessor.load_raw_train()

    def load_raw_doc(self):
        self.doc_docids, _, self.texts, _, _ = self.dataProcessor.load_raw_doc()

    def load_raw_dev(self):
        self.dev_docids, self.dev_questions, self.dev_answers, self.dev_answer_paragraphs, _ = self.dataProcessor.load_raw_dev()

    def load_raw_test(self):
        self.test_docids, self.test_questions, _, _, self.ids = self.dataProcessor.load_raw_test()

    def process_data(self):
        self.processed_qs = [self.dataProcessor.preprocess_questions_for_training(q) for q in self.questions]
        self.processed_doc = [self.dataProcessor.preprocess_doc_for_training(text) for text in self.texts]
        with open('processed_data.pkl', 'wb') as f:
            pickle.dump([self.processed_qs, self.processed_doc], f)

    def process_dev_data(self):
        self.dev_processed_qs = [self.dataProcessor.preprocess_questions_for_training(q) for q in self.dev_questions]

    def process_test_data(self):
        self.test_processed_qs = [self.dataProcessor.preprocess_questions_for_training(q) for q in self.test_questions]


    def build_training_data(self):
        len_all = len(self.processed_qs)
        correct = 0
        total = len(self.processed_qs)
        pars_vectors = []
        for i in range(len(self.processed_qs)):
            print i, ' / ', len_all
            qs = self.processed_qs[i]
            doc_id = self.docids[i]
            answer_par_id = self.answer_paragraphs[i]
            doc = self.processed_doc[doc_id]
            par_vectors = []
            for par in doc:
                embedding_par = self.dataProcessor.par_to_embedding(par)
                par_vectors.append(embedding_par)

            ranked_pars = self.dataProcessor.get_best_par(qs, doc)
            if ranked_pars:
                if len(ranked_pars) > self.config.MAX_PAIR_NUM:
                    ranked_pars = ranked_pars[:self.config.MAX_PAIR_NUM]
                # rank = 0
                for k, val in ranked_pars:
                    if k != answer_par_id:
                        label = 0
                    else:
                        label = 1
                        correct += 1
                    par_v = par_vectors[k]
                    par = doc[k]
                    paired_vec = self.dataProcessor.pair_to_vec_pre_sent(par_v, qs)
                    # concatenate bm25_score to the end of paired_vec
                    paired_vec += [val]
                    # rank += 1
                    paired_vec += self.dataProcessor.hand_crafted_features(par, qs)
                    self.training_vectors.append(paired_vec)
                    self.labels.append(label)

        pars_vectors.append(par_vectors)
        with open('pars_vectors.pkl', 'wb') as f:
            pickle.dump(pars_vectors, f)
        assert len(self.labels) == len(self.training_vectors)
        with open('training_vectors.pkl', 'wb') as f:
            pickle.dump(self.training_vectors, f)
        with open('labels.pkl', 'wb') as f:
            pickle.dump(self.labels, f)
        print len(self.labels)
        print len(self.training_vectors)
        print 100.0*correct/total

    def train_answer_paragraph_selection(self):
        self.clf = MLPClassifier(activation='tanh', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 50), random_state=1, shuffle=True)
        self.clf.fit(self.training_vectors, self.labels)
        with open(self.config.clf_save_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def evaluate(self):
        correct = 0
        count = 0
        len_all = len(self.dev_processed_qs)
        pars_vectors = []
        for i in range(len(self.dev_processed_qs)):
            print i, ' / ', len_all
            qs = self.dev_processed_qs[i]
            doc_id = self.dev_docids[i]
            answer_par_id = self.dev_answer_paragraphs[i]
            doc = self.processed_doc[doc_id]
            par_vectors = []
            for par in doc:
                embedding_par = self.dataProcessor.par_to_embedding(par)
                par_vectors.append(embedding_par)

            ranked_pars = self.dataProcessor.get_best_par(qs, doc)
            if ranked_pars:
                if len(ranked_pars) > self.config.MAX_PAIR_NUM:
                    ranked_pars = ranked_pars[:self.config.MAX_PAIR_NUM]
                # rank = 0
                for k, val in ranked_pars:
                    if k != answer_par_id:
                        label = 0
                    else:
                        label = 1
                        correct += 1
                    par_v = par_vectors[k]
                    par = doc[k]
                    paired_vec = self.dataProcessor.pair_to_vec_pre_sent(par_v, qs)
                    # concatenate bm25_score to the end of paired_vec
                    paired_vec += [val]
                    # rank += 1
                    paired_vec += self.dataProcessor.hand_crafted_features(par, qs)
                    self.training_vectors.append(paired_vec)
                    self.labels.append(label)

        pars_vectors.append(par_vectors)

        for i in range(len(self.dev_processed_qs)):
            print 'eval: ', i, ' / ', len_all
            qs = self.dev_processed_qs[i]
            doc_id = self.dev_docids[i]
            answer_par_id = self.dev_answer_paragraphs[i]
            doc = self.processed_doc[doc_id]

            wiki = self.dev_processed_wikis[i]
            # for h in range(len(wiki)):
            #     sent = wiki[h]
            #     for v in range(len(sent)):
            #         word = sent[v]
            #         if word.isdigit():
            #             wiki[h][v] = '-NUM-'
            inv_dict, _, _ = self.dataProcessor.build_inverted_indices(wiki)
            answer_inds = self.dev_answer_inds[i]
            sent_vectors = []
            for sent in wiki:
                embedding_sent = self.dataProcessor.sent_to_embedding(sent)
                sent_vectors.append(embedding_sent)
            for j in range(len(qs)):
                q = qs[j]
                ranked_docs = self.rank_qa_sents(q, inv_dict)
                ranked_dict = defaultdict(float)
                ranks = []
                if ranked_docs != -1:
                    for key, val in ranked_docs:
                        ranked_dict[key] = val
                    ranks = [x[0] for x in ranked_docs]
                q_vs = []
                answer_ind = answer_inds[j]
                for k in range(len(wiki)):
                    if k in ranks:
                        rank = ranks.index(k)
                    else:
                        rank = 100

                    sent_v = sent_vectors[k]
                    sent = wiki[k]
                    pair_v = self.dataProcessor.pair_to_vec_pre_sent(sent_v, q)
                    val = ranked_dict[k]
                    pair_v += [val]
                    pair_v += [rank]
                    pair_v += self.dataProcessor.hand_crafted_features(sent, q)
                    q_vs.append(pair_v)
                probs = self.clf.predict_proba(q_vs)
                # probs = []
                # for k in range(len(wiki)):
                    # probs.append(ranked_dict[k])
                max_prob_ind = np.argmax([x[1] for x in probs])
                # max_prob_ind = np.argmax(probs)
                if max_prob_ind == answer_ind:
                    correct += 1
                count += 1
        print correct*100.0/count, '%'
        print 'baseline: 59.1%'



if __name__ == '__main__':
    trainer = Trainer()
    # trainer.train_word2vec_model()
    # trainer = Trainer(mode)

