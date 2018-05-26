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
from sklearn import svm
import unicodecsv as csv
from file_loader import FileLoader
from data import Data
from basic_data_processor_sentence_embedding import BasicDataProcessor
from bm25 import BM25

class TrainBasedQA:
    def __init__(self):
        train = 1
        dev = 0
        test = 0
        load_processed_doc = 1
        load_doc_from_pkl = 0
        load_train_qs_from_pkl = 1
        load_dev_qs_from_pkl = 1
        load_test_qs_from_pkl = 1
        test_BM25 = 0
        train_sens_embedding = 1

        self.data = Data()
        self.config = Config()
        self.fileLoader = FileLoader(self.config, self.data)
        self.bdp = BasicDataProcessor(self.config, self.data)
        self.bm25 = BM25(self.config, self.data)
        # self.dataProcessor = DataProcessor()
        # self.dataProcessor.load_word2vec_model()

        self.fileLoader.load_doc()
        if load_processed_doc:
            if load_doc_from_pkl:
                with open(self.config.doc_processed_path, 'rb') as f:
                    self.data.doc_processed_sents_tokens, self.data.doc_original_sents_tokens, self.data.doc_ner_sents, self.data.doc_processed = pickle.load(
                        f)
            else:
                self.data.doc_processed_sents_tokens, self.data.doc_original_sents_tokens, self.data.doc_ner_sents, self.data.doc_processed = self.bdp.process_docs(
                    self.data.doc_texts)
                with open(self.config.doc_processed_path, 'wb') as f:
                    pickle.dump([self.data.doc_processed_sents_tokens, self.data.doc_original_sents_tokens,
                                 self.data.doc_ner_sents, self.data.doc_processed], f)

        if train:
            self.fileLoader.load_training_data()
            if load_train_qs_from_pkl:
                with open(self.config.train_qs_processed_path, 'rb') as f:
                    self.data.train_qs_processed = pickle.load(f)

            else:
                self.data.train_qs_processed = self.bdp.preprocess_questions(self.data.train_questions)
                with open(self.config.train_qs_processed_path, 'wb') as f:
                    pickle.dump(self.data.train_qs_processed, f)

            if test_BM25:
                self.bm25.test_training_BM25_accuracy(10)

            if train_sens_embedding:
                self.bdp.train_sens_embeddings()

        if dev:
            self.fileLoader.load_dev_data()
            if load_dev_qs_from_pkl:
                with open(self.config.dev_qs_processed_path, 'rb') as f:
                    self.data.dev_qs_processed = pickle.load(f)
            else:
                self.data.dev_qs_processed = self.bdp.preprocess_questions(self.data.dev_questions)
                with open(self.config.dev_qs_processed_path, 'wb') as f:
                    pickle.dump(self.data.dev_qs_processed, f)
            if test_BM25:
                self.bm25.test_dev_BM25_accuracy(10)

            self.answer_dev()

        if test:
            self.fileLoader.load_test_data()
            if load_test_qs_from_pkl:
                with open(self.config.test_qs_processed_path, 'rb') as f:
                    self.data.test_qs_processed = pickle.load(f)
            else:
                self.data.test_qs_processed = self.bdp.preprocess_questions(self.data.test_questions)
                with open(self.config.test_qs_processed_path, 'wb') as f:
                    pickle.dump(self.data.test_qs_processed, f)



        # one dimensional vector of [paragraph_embedding , query_embedding, bm25_score, hand_crafted_features]
        # self.training_vectors = []
        # 1 for correct paragraph and 0 for wrong paragraph
        # self.labels = []
        # if load_model:
        #     with open('clf_withFeatures.pkl', 'rb') as f:
        #         self.clf = pickle.load(f)
        # else:
        #     self.build_training_data()
        #     self.train_answer_paragraph_selection()
            # self.train_answer_paragraph_selection_SVM()

        # self.evaluate()
        # self.evaluateTrain()


    def build_training_data(self):
        correct = 0
        train_num = 1
        total = train_num
        # total = len(self.processed_qs)
        pars_vectors = []
        for i in range(train_num):
        # for i in range(len(self.processed_qs)):
            print(i, ' / ', total)
            qs = self.processed_qs[i]
            doc_id = self.docids[i]
            answer_par_id = self.answer_paragraphs[i]
            doc = self.processed_doc[doc_id]
            par_vectors = []
            for par in doc:
                embedding_par = self.dataProcessor.par_to_embedding(par)
                par_vectors.append(embedding_par)

            print("doc :", len(doc))
            ranked_pars = self.dataProcessor.get_best_par(qs, doc)
            print("ranked_pars :", len(ranked_pars))
            if ranked_pars:
                # if len(ranked_pars) > self.config.MAX_PAIR_NUM:
                #     ranked_pars = ranked_pars[:self.config.MAX_PAIR_NUM]

                count_N = -1
                for k, val in ranked_pars:
                    count_N += 1
                    if k != answer_par_id:
                        label = 0
                        if count_N > 1 and count_N < len(ranked_pars) - 6:
                            continue
                    else:
                        label = 1
                        print(answer_par_id)
                    if count_N == 0:
                        correct += 1
                    par_v = par_vectors[k]
                    par = doc[k]
                    paired_vec = []
                    paired_vec = self.dataProcessor.pair_to_vec_pre_sent(par_v, qs)
                    # concatenate bm25_score to the end of paired_vec
                    paired_vec += [val]
                    # rank += 1
                    paired_vec += self.dataProcessor.hand_crafted_features(par, qs)
                    self.training_vectors.append(paired_vec)
                    # print k, answer_par_id, label
                    self.labels.append(label)

                # for k, val in ranked_pars:
                #     if k != answer_par_id:
                #         label = 0
                #     else:
                #         label = 1
                #         correct += 1
                #     par_v = par_vectors[k]
                #     par = doc[k]
                #     paired_vec = self.dataProcessor.pair_to_vec_pre_sent(par_v, qs)
                #     # concatenate bm25_score to the end of paired_vec
                #     paired_vec += [val]
                #     # rank += 1
                #     paired_vec += self.dataProcessor.hand_crafted_features(par, qs)
                #     self.training_vectors.append(paired_vec)
                #     self.labels.append(label)

        pars_vectors.append(par_vectors)
        # with open('pars_vectors.pkl', 'wb') as f:
        #     pickle.dump(pars_vectors, f)
        # with open('training_vectors1.pkl', 'wb') as f:
        #     pickle.dump(self.training_vectors, f)
        # with open('labels1.pkl', 'wb') as f:
        #     pickle.dump(self.labels, f)
        assert len(self.labels) == len(self.training_vectors)
        # print len(self.labels)
        # print len(self.training_vectors)

        print(self.training_vectors)
        print(self.labels)
        print("answer_par_id :", answer_par_id)

        print(100.0*correct/total)

    def train_answer_paragraph_selection(self):
        self.clf = MLPClassifier(activation='relu', alpha=1e-5, hidden_layer_sizes=(200, 50),verbose=1, random_state=1, shuffle=True)
        self.clf.fit(self.training_vectors, self.labels)
        with open(self.config.clf_save_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def train_answer_paragraph_selection_SVM(self):
        self.clf = svm.SVC(probability=True)
        self.clf.fit(self.training_vectors, self.labels)
        with open(self.config.clf_save_path, 'wb') as f:
            pickle.dump(self.clf, f)

    def evaluate(self):
        # with open('clf_withFeatures.pkl', 'rb') as f:
        #     self.clf = pickle.load(f)
        # with open('training_vectors.pkl', 'rb') as f:
        #     q_vectors = pickle.load(f)
        # probs = self.clf.predict_proba(q_vectors)
        # print probs
        # max_prob_ind = np.argmax([x[1] for x in probs])
        # print max_prob_ind

        len_all = len(self.dev_processed_qs)
        correct = 0
        for i in range(len(self.dev_processed_qs)):
            print(i, ' / ', len_all)
            qs = self.dev_processed_qs[i]
            doc_id = self.dev_docids[i]
            answer_par_id = self.dev_answer_paragraphs[i]
            doc = self.processed_doc[doc_id]
            par_vectors = []
            # for par in doc:
            #     embedding_par = self.dataProcessor.par_to_embedding(par)
            #     par_vectors.append(embedding_par)

            ranked_pars = self.dataProcessor.get_best_par(qs, doc)
            if ranked_pars:
                vectors = []
                # if len(ranked_pars) > self.config.MAX_PAIR_NUM:
                #     ranked_pars = ranked_pars[:self.config.MAX_PAIR_NUM]
                for k, val in ranked_pars:
                    if k == answer_par_id:
                        correct += 1
                    break
                #     par_v = par_vectors[k]
                #     par = doc[k]
                #     paired_vec = self.dataProcessor.pair_to_vec_pre_sent(par_v, qs)
                #     # concatenate bm25_score to the end of paired_vec
                #     paired_vec += [val]
                #     # rank += 1
                #     paired_vec += self.dataProcessor.hand_crafted_features(par, qs)
                #     vectors.append(paired_vec)
                #
                # probs = self.clf.predict_proba(vectors)
                # max_prob_ind = np.argmax([x[1] for x in probs])
                # # max_prob_ind = np.argmax(probs)
                # # print len(probs)
                # # print max_prob_ind, answer_par_id
                # if max_prob_ind == answer_par_id:
                #     correct += 1
                #     # print correct

        print(correct*100.0/len_all, '%')
        # print 'baseline: 59.1%'

    def evaluateTrain(self):
        # with open('clf_withFeatures.pkl', 'rb') as f:
        #     self.clf = pickle.load(f)
        # with open('training_vectors.pkl', 'rb') as f:
        #     q_vectors = pickle.load(f)
        # probs = self.clf.predict_proba(q_vectors)
        # print probs
        # max_prob_ind = np.argmax([x[1] for x in probs])
        # print max_prob_ind

        train_num = 1
        len_all = train_num
        # len_all = len(self.processed_qs)
        correct = 0
        for i in range(train_num):
        # for i in range(len(self.processed_qs)):
            print(i, ' / ', len_all)
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
                # print ranked_pars[0][0], answer_par_id
                vectors = []
                # if len(ranked_pars) > self.config.MAX_PAIR_NUM:
                #     ranked_pars = ranked_pars[:self.config.MAX_PAIR_NUM]
                for k, val in ranked_pars:
                    par_v = par_vectors[k]
                    par = doc[k]
                    paired_vec = []
                    paired_vec = self.dataProcessor.pair_to_vec_pre_sent(par_v, qs)
                    # concatenate bm25_score to the end of paired_vec
                    paired_vec += [val]
                    # rank += 1
                    paired_vec += self.dataProcessor.hand_crafted_features(par, qs)
                    vectors.append(paired_vec)

                print("vectors :", vectors)
                probs = self.clf.predict_proba(vectors)
                max_prob_ind = np.argmax([x[1] for x in probs])
                # max_prob_ind = np.argmax(probs)
                # print len(probs)
                print(max_prob_ind, answer_par_id)
                if max_prob_ind == answer_par_id:
                    correct += 1
                    print(correct)

        print(correct * 100.0 / len_all, '%')
        print('baseline: 59.1%')

if __name__ == '__main__':
    train_based_QA = TrainBasedQA()

