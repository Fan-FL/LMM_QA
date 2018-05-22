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

class Trainer:
    def __init__(self, train=1, dev=1, test=None, load_data=1, load_model=None):
        self.dataProcessor = DataProcessor()
        self.config = Config()
        self.fileLoader = FileLoader(self.config)
        self.dataProcessor.load_word2vec_model()

        self.load_raw_doc()

        if train == 1:
            self.load_raw_training()

        if dev == 1:
            self.load_raw_dev()
            self.process_dev_data()

        if test == 1:
            self.load_raw_test()
            self.process_test_data()

        if load_data:
            with open('processed_data.pkl', 'rb') as f:
                self.processed_qs, self.processed_doc = pickle.load(f)
        else:
            self.process_data()

        # self.test_training_BM25_accuracy(10)
        # self.test_dev_BM25_accuracy(10)



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
        correct = 0
        train_num = 1
        total = train_num
        # total = len(self.processed_qs)
        pars_vectors = []
        for i in range(train_num):
        # for i in range(len(self.processed_qs)):
            print i, ' / ', total
            qs = self.processed_qs[i]
            doc_id = self.docids[i]
            answer_par_id = self.answer_paragraphs[i]
            doc = self.processed_doc[doc_id]
            par_vectors = []
            for par in doc:
                embedding_par = self.dataProcessor.par_to_embedding(par)
                par_vectors.append(embedding_par)

            print "doc :", len(doc)
            ranked_pars = self.dataProcessor.get_best_par(qs, doc)
            print "ranked_pars :", len(ranked_pars)
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
                        print answer_par_id
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

        print self.training_vectors
        print self.labels
        print "answer_par_id :", answer_par_id

        print 100.0*correct/total

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
            print i, ' / ', len_all
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

        print correct*100.0/len_all, '%'
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

                print "vectors :", vectors
                probs = self.clf.predict_proba(vectors)
                max_prob_ind = np.argmax([x[1] for x in probs])
                # max_prob_ind = np.argmax(probs)
                # print len(probs)
                print max_prob_ind, answer_par_id
                if max_prob_ind == answer_par_id:
                    correct += 1
                    print correct

        print correct * 100.0 / len_all, '%'
        print 'baseline: 59.1%'

    def test_training_BM25_accuracy(self, max_tolerant_num):
        n_accuary = defaultdict(int)
        # total = 10
        total = len(self.processed_qs)
        pars_vectors = []
        # for i in range(total):
        for i in range(len(self.processed_qs)):
            print i, ' / ', total
            qs = self.processed_qs[i]
            doc_id = self.docids[i]
            answer_par_id = self.answer_paragraphs[i]
            doc = self.processed_doc[doc_id]

            ranked_pars = self.dataProcessor.get_best_par(qs, doc)

            if ranked_pars:
                count_N = -1
                # print [k for k, v in ranked_pars]
                # print answer_par_id
                for k, val in ranked_pars:
                    count_N += 1
                    if count_N < max_tolerant_num:
                        if k == answer_par_id:
                            for m in range(max_tolerant_num, count_N, -1):
                                n_accuary[m] += 1
                    else:
                        break

        # print n_accuary
        with open('training_BM25_accuracy.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['total', str(total)])
            csv_writer.writerow(['N', "correct", 'accuracy'])
            for k, v in n_accuary.items():
                csv_writer.writerow([str(k), str(v), str(1.0*v/total)])

    def test_dev_BM25_accuracy(self, max_tolerant_num):
        n_accuary = defaultdict(int)
        # total = 10
        total = len(self.dev_processed_qs)
        pars_vectors = []
        # for i in range(total):
        for i in range(len(self.dev_processed_qs)):
            print i, ' / ', total
            qs = self.dev_processed_qs[i]
            doc_id = self.dev_docids[i]
            answer_par_id = self.dev_answer_paragraphs[i]
            doc = self.processed_doc[doc_id]

            ranked_pars = self.dataProcessor.get_best_par(qs, doc)

            if ranked_pars:
                count_N = -1
                # print [k for k, v in ranked_pars]
                # print answer_par_id
                for k, val in ranked_pars:
                    count_N += 1
                    if count_N < max_tolerant_num:
                        if k == answer_par_id:
                            for m in range(max_tolerant_num, count_N, -1):
                                n_accuary[m] += 1
                    else:
                        break

        # print n_accuary
        with open('dev_BM25_accuracy.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['total', str(total)])
            csv_writer.writerow(['N', "correct", 'accuracy'])
            for k, v in n_accuary.items():
                csv_writer.writerow([str(k), str(v), str(1.0*v/total)])

if __name__ == '__main__':
    trainer = Trainer()
    # with open('training_vectors.pkl', 'rb') as f:
    #     x = pickle.load(f)
    # with open('training_vectors2.pkl', 'rb') as f:
    #     y = pickle.load(f)
    # x += y
    # with open('training_vectors3.pkl', 'wb') as f:
    #     pickle.dump(x, f)

    # with open('training_vectors3.pkl', 'rb') as f:
    #     x = pickle.load(f)
    #     for a in x:
    #         print a
    # trainer.train_word2vec_model()
    # trainer = Trainer(mode)

