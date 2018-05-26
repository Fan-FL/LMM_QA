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
from nltk import word_tokenize
from nltk import pos_tag

class RuleBasedQA:
    def __init__(self):
        train = 0
        dev = 1
        test = 0
        load_processed_doc = 1
        load_doc_from_pkl = 1
        load_train_qs_from_pkl = 1
        load_dev_qs_from_pkl = 1
        load_test_qs_from_pkl = 1
        test_BM25 = 0
        train_sens_embedding = 0

        self.data = Data()
        self.config = Config()
        self.fileLoader = FileLoader(self.config, self.data)
        self.bdp = BasicDataProcessor(self.config, self.data)
        self.bm25 = BM25(self.config, self.data)
        # self.dataProcessor = DataProcessor()
        # self.dataProcessor.load_word2vec_model()

        self.fileLoader.load_doc()

        # if load_processed_doc:
        #     if load_doc_from_pkl:
        #         with open(self.config.doc_processed_path, 'rb') as f:
        #             self.data.doc_processed_sents_tokens, self.data.doc_original_sents_tokens, self.data.doc_ner_sents, self.data.doc_processed = pickle.load(f)
        #     else:
        #         self.data.doc_processed_sents_tokens, self.data.doc_original_sents_tokens, self.data.doc_ner_sents, self.data.doc_processed = self.bdp.process_docs(self.data.doc_texts)
        #         with open(self.config.doc_processed_path, 'wb') as f:
        #             pickle.dump([self.data.doc_processed_sents_tokens, self.data.doc_original_sents_tokens, self.data.doc_ner_sents, self.data.doc_processed], f)

        if load_processed_doc:
            if load_doc_from_pkl:
                with open(self.config.doc_processed_path_bak, 'rb') as f:
                    self.data.doc_processed = pickle.load(f)
            else:
                self.data.doc_processed = self.bdp.process_docs(self.data.doc_texts)
                with open(self.config.doc_processed_path_bak, 'wb') as f:
                    pickle.dump(self.data.doc_processed, f)


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

            self.answer_train()

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


    def answer_train(self, save=True):
        correct = 0
        correct_id = 0
        total = len(self.data.train_qs_processed)
        with open('train.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['W/R', 'query', 'predicted_id', 'actual_id', 'predicted_answer', 'actual_answer', 'predicted_answer_type'])
            # for i in range(20):
            for i in range(100):
            # for i in range(len(self.data.train_questions)):
            #     print(i, " / ", len(self.data.train_questions))

                train_qs = self.data.train_questions[i]
                doc_id = self.data.train_doc_ids[i]
                train_doc = self.data.doc_texts[doc_id]
                train_answer = self.data.train_answers[i]
                train_answer_par_id = self.data.train_answer_par_ids[i]
                train_qs_processed = self.data.train_qs_processed[i]
                doc_processed = self.data.doc_processed[doc_id]
                doc_ner = self.data.doc_ner_sents[doc_id]
                doc_original_tokens = self.data.doc_original_sents_tokens[doc_id]
                doc_ners = []

                # load_train_rule_pkl = 0
                # if load_train_rule_pkl:
                #     with open(self.config.train_rule_pkl, 'rb') as f:
                #         doc_ners, doc_original_tokens = pickle.load(f)
                #
                # else:
                #     doc_ners, doc_original_tokens = self.preprocess_doc_for_rule(train_doc)
                #     with open(self.config.train_rule_pkl, 'wb') as f:
                #         pickle.dump([doc_ners, doc_original_tokens], f)

                wh = self.extract_wh_word(train_qs_processed)
                possible_qs_type_rank, qs_type = self.identify_question_type(wh, train_qs_processed)
                answer = 'unknown'
                answer_types = []
                pred_par_id = -1
                if possible_qs_type_rank:
                    bm25_rank = self.bm25.sort_by_bm25_score(train_qs_processed, doc_processed)
                    bm25_rank_par_ids = [x[0] for x in bm25_rank]
                    for par_id in bm25_rank_par_ids:
                        par_answer, par_answer_types, par_pred_par_id = self.pred_answer_type_id(doc_ner, doc_original_tokens,
                            par_id, train_qs_processed,
                                         possible_qs_type_rank, qs_type)
                        if par_answer != -1:
                            answer = par_answer
                            answer_types = par_answer_types
                            pred_par_id = par_pred_par_id
                            break
                types = ' '.join(answer_types)
                if pred_par_id == int(train_answer_par_id):
                    correct_id += 1
                if train_answer == answer:
                    csv_writer.writerow(["##right##", train_qs, pred_par_id, train_answer_par_id, answer, train_answer, types])
                    correct += 1
                else:
                    csv_writer.writerow(["##wrong##", train_qs, pred_par_id, train_answer_par_id, answer,  train_answer, types])
                print(train_answer, " ; ", answer)
                # print "correct :", correct
            csv_writer.writerow([str(correct)])
            csv_writer.writerow([str(correct_id)])
            csv_writer.writerow([str(total)])
        print(correct*100.0/total)
        print(correct_id*100.0/total)

    def answer_dev(self, save=True):
        correct = 0
        correct_id = 0
        total = len(self.data.dev_qs_processed)
        with open('dev.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['W/R', 'query', 'predicted_id', 'actual_id', 'predicted_answer', 'actual_answer',
                                 'predicted_answer_type'])
            # for i in range(20):
            for i in range(5):
                # for i in range(len(self.data.dev_questions)):
                #     print(i, " / ", len(self.data.dev_questions))

                dev_qs = self.data.dev_questions[i]
                doc_id = self.data.dev_doc_ids[i]
                # dev_doc = self.data.doc_texts[doc_id]
                dev_answer = self.data.dev_answers[i]
                dev_answer_par_id = self.data.dev_answer_par_ids[i]
                dev_qs_processed = self.data.dev_qs_processed[i]
                doc_processed = self.data.doc_processed[doc_id]
                doc_ner = self.data.doc_ner_sents[doc_id]
                doc_original_tokens = self.data.doc_original_sents_tokens[doc_id]
                doc_ners = []

                # load_dev_rule_pkl = 0
                # if load_dev_rule_pkl:
                #     with open(self.config.dev_rule_pkl, 'rb') as f:
                #         doc_ners, doc_original_tokens = pickle.load(f)
                #
                # else:
                #     doc_ners, doc_original_tokens = self.preprocess_doc_for_rule(dev_doc)
                #     with open(self.config.dev_rule_pkl, 'wb') as f:
                #         pickle.dump([doc_ners, doc_original_tokens], f)

                wh = self.extract_wh_word(dev_qs_processed)
                possible_qs_type_rank, qs_type = self.identify_question_type(wh, dev_qs_processed)
                answer = 'unknown'
                answer_types = []
                pred_par_id = -1
                if possible_qs_type_rank:
                    bm25_rank = self.bm25.sort_by_bm25_score(dev_qs_processed, doc_processed)
                    bm25_rank_par_ids = [x[0] for x in bm25_rank]
                    for par_id in bm25_rank_par_ids:
                        par_answer, par_answer_types, par_pred_par_id = self.pred_answer_type_id(doc_ner,
                                                                                                 doc_original_tokens,
                                                                                                 par_id,
                                                                                                 dev_qs_processed,
                                                                                                 possible_qs_type_rank,
                                                                                                 qs_type)
                        if par_answer != -1:
                            answer = par_answer
                            answer_types = par_answer_types
                            pred_par_id = par_pred_par_id
                            break
                types = ' '.join(answer_types)
                if pred_par_id == int(dev_answer_par_id):
                    correct_id += 1
                if dev_answer == answer:
                    csv_writer.writerow(
                        ["##right##", dev_qs, pred_par_id, dev_answer_par_id, answer, dev_answer, types])
                    correct += 1
                else:
                    csv_writer.writerow(
                        ["##wrong##", dev_qs, pred_par_id, dev_answer_par_id, answer, dev_answer, types])
                print(dev_answer, " ; ", answer)
                # print "correct :", correct
            csv_writer.writerow([str(correct)])
            csv_writer.writerow([str(correct_id)])
            csv_writer.writerow([str(total)])
        print(correct * 100.0 / total)
        print(correct_id * 100.0 / total)


    def preprocess_doc_for_rule(self, doc):
        normal_tokens = [word_tokenize(par.replace("\u200b",'').replace("\u2014",'')) for par in doc]
        remove_punc_tokens = [[token for token in tokens if not self.bdp.is_pure_puncs(token)] for tokens in normal_tokens]
        remove_punc_in_tokens = [[self.bdp.remove_punc_in_token(token) for token in tokens] for tokens in remove_punc_tokens]
        ners = self.bdp.ner_tagging(remove_punc_in_tokens)
        return ners, remove_punc_in_tokens

    def mark_wh_word(self, words):
        wh_word_marks = [0] * (len(self.config.WH_words) + 1)
        for word in words:
            if word in self.config.WH_words:
                wh_word_marks[self.config.WH_words.index(word)] = 1
                return wh_word_marks
        wh_word_marks[-1] = 1
        return wh_word_marks

    def extract_wh_word(self, words):
        for word in words:
            if word.lower() in self.config.WH_words or word.lower() == 'whom':
                return word
        return -1

    def identify_question_type(self, wh, q_words):
        lower = self.bdp.lower_tokens(q_words)
        # open_words = self.dataProcessor.remove_stop_words(lower)
        raw_q_sent = ' '.join(lower)
        if wh == 'what':
            if 'length' in raw_q_sent:
                return ['NUMBER'], 'length'
            if 'what year' in raw_q_sent:
                return ['NUMBER'], 'year'
            if 'date' in raw_q_sent:
                return ['NUMBER'], 'time'
            if 'percent' in raw_q_sent:
                return ['NUMBER'], 'percentage'
            if 'number' in raw_q_sent:
                return ['NUMBER'], 'number'
            if 'place' in raw_q_sent or 'country' in raw_q_sent or 'city' in raw_q_sent or 'locat' in raw_q_sent or 'site' in raw_q_sent:
                return ['LOCATION', 'OTHER', 'O', 'NUMBER', 'PERSON'], 'place'
            else:
                return ['OTHER', 'O', 'PERSON', 'LOCATION', 'NUMBER'], 'else'
        elif wh == 'when':
            return ['NUMBER'], 'time'
        elif wh == 'who' or wh == 'whom':
            return ['PERSON'], 'person'
        elif wh == 'where':
            return ['LOCATION', 'OTHER', 'O'], 'location'
        elif wh == 'how':
            if 'how long' in raw_q_sent or 'how far' in raw_q_sent or 'how fast' in raw_q_sent:
                return ['NUMBER'], 'length'
            elif 'how many' in raw_q_sent:
                return ['NUMBER'], 'number'
            elif 'how much' in raw_q_sent:
                return ['NUMBER'], 'money'
            else:
                return ['OTHER', 'NUMBER', 'LOCATION', 'PERSON'], 'else'
        elif wh == 'which':
            if 'which year' in raw_q_sent:
                return ['NUMBER'], 'year'
            if 'place' in raw_q_sent or 'country' in raw_q_sent or 'city' in raw_q_sent or 'locat' in raw_q_sent or 'site' in raw_q_sent:
                return ['LOCATION', 'OTHER', 'O', 'NUMBER', 'PERSON'], 'place'
            if 'person' in raw_q_sent:
                return ['PERSON', 'OTHER', 'O', 'LOCATION', 'NUMBER'], 'person'
            else:
                return ['OTHER', 'O', 'LOCATION', 'PERSON', 'NUMBER'], 'else'
        else:
            return ['OTHER', 'O', 'LOCATION', 'PERSON', 'NUMBER'], 'else'

    def pred_answer_type_id(self, pars_ner, pars_original_tokens, par_id, qs_processed,
                                         possible_qs_type_rank, qs_type):
        ner_par = []
        for sent in pars_ner[par_id]:
            ner_par += sent
        original_par = []
        for sent in pars_original_tokens[par_id]:
            original_par += sent
        entities = self.get_combined_entities(ner_par)
        assert len(original_par) == len(ner_par)
        # entities = self.remove_entity_in_qs(qs_processed, entities)
        ner_type_to_entities_dict = self.get_ner_type_to_entities_dict(entities)
        grouped_entities_strings_lemmatized = [self.bdp.lemmatize(tup[0]) for tup in entities]
        for type in possible_qs_type_rank:
            if len(ner_type_to_entities_dict[type]) != 0:
                assert ner_type_to_entities_dict[type]
                one_type_entities = ner_type_to_entities_dict[type]
                one_type_grouped_entities_strings = [x[0] for x in one_type_entities]
                one_type_grouped_entities_strings = self.bdp.remove_stop_words(one_type_grouped_entities_strings)
                if type == 'O':
                    one_type_grouped_entities_strings = [x[0] for x in pos_tag(one_type_grouped_entities_strings)
                                                        if 'NN' in x[1]]
                distance = []
                possible_entity_pos = []

                qs_token_in_entity_pos = []
                for qs_token in qs_processed:
                    for i in range(len(grouped_entities_strings_lemmatized)):
                        entity_string = grouped_entities_strings_lemmatized[i]
                        if qs_token in entity_string:
                            qs_token_in_entity_pos.append(i)

                for entity in one_type_grouped_entities_strings:
                    for j in range(len(entities)):
                        word = entities[j][0]
                        if word == entity:
                            sum_dist = 0
                            for k in qs_token_in_entity_pos:
                                sum_dist += (abs(j - k))
                            distance.append(sum_dist)
                            possible_entity_pos.append(j)
                            break
                assert len(possible_entity_pos) == len(distance)
                if distance:
                    min_idx = np.argmin(distance)

                    best_entity = entities[possible_entity_pos[min_idx]][0]
                    if qs_type == 'year' and type == 'NUMBER':
                        years = [x for x in one_type_grouped_entities_strings if len(x) == 4]
                        if len(best_entity) != 4 and years:
                            best_entity = years[0]
                            return best_entity, possible_qs_type_rank, par_id

                    chunk_entities = entities[:possible_entity_pos[min_idx]]
                    actual_idx = 0
                    for entity in chunk_entities:
                          actual_idx += len(entity[0].split())
                    actual_end_idx = actual_idx + len(best_entity.split())
                    if qs_type == 'length' and actual_end_idx < len(original_par):
                        actual_end_idx += 1
                    # if type=='NUMBER' and qtype=='length':
                    #     if actual_end_ind < len(raw_doc):
                    #         next_word = raw_doc[actual_end_ind]
                    #         synsets = wordnet.synsets(next_word)
                    #         if synsets:
                    #             best_syn = synsets[0]
                    #             hypers = best_syn.hypernyms()
                    #             if hypers:
                    #                 best_hyper = hypers[0]
                    #                 lemmas = best_hyper.lemma_names()
                    #                 if lemmas:
                    #                     lemma = lemmas[0]
                    #                     if 'unit' in lemma:
                    #                         actual_end_ind += 1
                    #

                    actual_best_entity = ' '.join(original_par[actual_idx:actual_end_idx])
                    actual_best_entity = actual_best_entity.replace('\"', '')
                    actual_best_entity = actual_best_entity.replace(',', '-COMMA-')
                    if qs_type == 'percentage' and 'per' not in actual_best_entity and actual_best_entity[-1].isdigit():
                        if actual_end_idx < len(raw_doc):
                            if original_par[actual_end_idx] == 'per':
                                actual_best_entity += ' per cent'
                            elif original_par[actual_end_idx] == 'percent':
                                actual_best_entity += ' percent'
                            else:
                                actual_best_entity += '%'
                        else:
                            actual_best_entity += '%'
                    if qs_type == 'money' and actual_best_entity[0].isdigit():
                        actual_best_entity = '$' + actual_best_entity

                    return actual_best_entity, possible_qs_type_rank, par_id
                return -1, [], par_id

    def get_combined_entities(self, ner_par):
        entities = []
        ner_group = []
        prev_ner_type = ''
        for ner_tuple in ner_par:
            current_ner_type = ner_tuple[1]
            if not prev_ner_type:
                ner_group.append(ner_tuple)
                prev_ner_type = current_ner_type
            else:
                if current_ner_type == prev_ner_type:
                    ner_group.append(ner_tuple)
                else:
                    entities += self.process_combined_entity(ner_group, prev_ner_type)
                    ner_group = [ner_tuple]
                    prev_ner_type = current_ner_type
        entities += self.process_combined_entity(ner_group, prev_ner_type)
        return entities

    def process_combined_entity(self, ner_group, ner_type):
        entities = []
        if ner_type == 'O':
            for ner_tuple in ner_group:
                entities.append(ner_tuple)
        else:
            entity = [ner_tuple[0] for ner_tuple in ner_group]
            entity_item = (' '.join(entity), ner_type)
            entities.append(entity_item)
        return entities

    def remove_entity_in_qs(self, qs_processed, entities):
        valid_entities = []
        for entity in entities:
            entity_words = entity[0].split()
            for word in entity_words:
                if self.bdp.lemmatize(word) not in qs_processed:
                    valid_entities.append(entity)
                    break
        return valid_entities

    def get_ner_type_to_entities_dict(self, entities):
        ner_type_to_entities_dict = defaultdict(list)
        for entity in entities:
            ner_type = entity[1]
            ner_type_to_entities_dict[ner_type].append(entity)
        return ner_type_to_entities_dict

if __name__ == '__main__':
    rule_based_QA = RuleBasedQA()