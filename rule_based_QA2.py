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
from basic_data_processor import BasicDataProcessor
from bm25 import BM25
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

class RuleBasedQA:
    def __init__(self):

        train = 0
        dev = 0
        test = 1
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

        self.location = ['LOCATION']
        self.city = ['CITY']
        self.state_or_province = ['STATE_OR_PROVINCE']
        self.country = ['COUNTRY']
        self.nationality = ['NATIONALITY']
        self.religion = ['RELIGION']
        self.person = ['PERSON']
        self.organization = ['ORGANIZATION']
        self.number = ["NUMBER"]
        self.ordinal = ['ORDINAL']
        self.money = ["MONEY"]
        self.percent = ["PERCENT"]
        self.date = ['DATE']
        self.time = ['TIME']
        self.duration = ['DURATION']
        self.cause_of_death = ['CAUSE_OF_DEATH']
        self.ideology = ['IDEOLOGY']
        self.other = ['SET', "MISC", 'EMAIL', 'URL', 'TITLE', 'IDEOLOGY', 'CRIMINAL_CHARGE']

        # self.dataProcessor = DataProcessor()
        # self.dataProcessor.load_word2vec_model()

        # print(self.preprocess_questions("What location"))

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

            self.answer_test()

    def answer_dev(self, save=True):
        correct = 0
        correct_id = 0
        total = int(len(self.data.dev_questions))
        # total = 6
        doc_temp = {}

        with open('dev.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['W/R', 'query', 'predicted_id', 'actual_id', 'predicted_answer', 'actual_answer',
                                 'predicted_answer_type', 'predicated_candidates'])
            # for i in range(total):
            for i in range(96,97):
                print(i, " / ", total)

                dev_qs = self.data.dev_questions[i]
                doc_id = self.data.dev_doc_ids[i]
                dev_doc = self.data.doc_texts[doc_id]
                dev_answer = self.data.dev_answers[i]
                dev_answer_par_id = self.data.dev_answer_par_ids[i]
                dev_qs_processed = self.preprocess_questions(dev_qs)
                doc_processed = self.data.doc_processed[doc_id]

                doc_entities = []
                if doc_id in doc_temp:
                    doc_entities = doc_temp[doc_id]
                else:
                    doc_entities = []
                    for par in dev_doc:
                        doc_entities.append(self.ner_process(par))
                    doc_temp[doc_id] = doc_entities

                wh = self.extract_wh_word(dev_qs_processed)
                possible_qs_type_rank, qs_type = self.identify_question_type(wh, dev_qs_processed)
                answer = 'unknown'
                answer_types = []
                pred_par_id = -1
                candidate_answers = ''
                if possible_qs_type_rank:
                    bm25_rank = self.bm25.sort_by_bm25_score(dev_qs_processed, doc_processed)
                    bm25_rank_par_ids = [x[0] for x in bm25_rank]
                    for par_id in bm25_rank_par_ids:
                        # print("par_id", par_id)
                        par_answer, par_answer_types, par_candidate_answers = self.pred_answer_type(par_id, doc_entities[par_id],
                                                                                                    dev_qs_processed,
                                                                            possible_qs_type_rank,
                                                                            qs_type)
                        if par_answer != -1:
                            answer = par_answer
                            answer_types = par_answer_types
                            pred_par_id = par_id
                            candidate_answers = '; '.join(par_candidate_answers)
                            break
                types = ' '.join(answer_types)
                if pred_par_id == int(dev_answer_par_id):
                    correct_id += 1
                if dev_answer == answer:
                    csv_writer.writerow(
                        ["##right##", dev_qs, pred_par_id, dev_answer_par_id, answer, dev_answer, types, candidate_answers])
                    correct += 1
                else:
                    csv_writer.writerow(
                        ["##wrong##", dev_qs, pred_par_id, dev_answer_par_id, answer, dev_answer, types, candidate_answers])
                print(dev_answer, " ; ", answer)
                # print "correct :", correct
            csv_writer.writerow([str(correct), str(correct * 100.0 / total)])
            csv_writer.writerow([str(correct_id), str(correct_id * 100.0 / total)])
            csv_writer.writerow([str(total)])
        print(correct * 100.0 / total)
        print(correct_id * 100.0 / total)
        print("best : 12.399095899257345")

    def answer_test(self, save=True):
        correct = 0
        correct_id = 0
        total = int(len(self.data.test_questions))
        # total = 6
        doc_temp = {}

        with open('test_results.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['id', 'answer'])
            for i in range(total):
            # for i in range(30):
                print(i, " / ", total)

                test_qs = self.data.test_questions[i]
                doc_id = self.data.test_doc_ids[i]
                test_doc = self.data.doc_texts[doc_id]
                test_qs_processed = self.preprocess_questions(test_qs)
                doc_processed = self.data.doc_processed[doc_id]
                test_id = self.data.test_ids[i]

                doc_entities = []
                if doc_id in doc_temp:
                    doc_entities = doc_temp[doc_id]
                else:
                    doc_entities = []
                    for par in test_doc:
                        doc_entities.append(self.ner_process(par))
                    doc_temp[doc_id] = doc_entities

                wh = self.extract_wh_word(test_qs_processed)
                possible_qs_type_rank, qs_type = self.identify_question_type(wh, test_qs_processed)
                answer = 'unknown'
                answer_types = []
                pred_par_id = -1
                candidate_answers = ''
                if possible_qs_type_rank:
                    bm25_rank = self.bm25.sort_by_bm25_score(test_qs_processed, doc_processed)
                    bm25_rank_par_ids = [x[0] for x in bm25_rank]
                    # print(bm25_rank_par_ids)
                    for par_id in bm25_rank_par_ids:
                        # print("par_id", par_id)
                        par_answer, par_answer_types, par_candidate_answers = self.pred_answer_type(par_id, doc_entities[par_id],
                                                                                                    test_qs_processed,
                                                                            possible_qs_type_rank,
                                                                            qs_type)
                        if par_answer != -1:
                            answer = par_answer
                            answer_types = par_answer_types
                            pred_par_id = par_id
                            candidate_answers = '; '.join(par_candidate_answers)
                            break
                csv_writer.writerow([test_id, answer])
        # print(correct * 100.0 / total)
        # print(correct_id * 100.0 / total)
        # print("best : 12.399095899257345")

    def preprocess_doc_for_rule(self, doc):
        normal_tokens = [word_tokenize(par.replace("\u200b",'').replace("\u2014",'')) for par in doc]
        remove_punc_tokens = [[token for token in tokens if not self.bdp.is_pure_puncs(token)] for tokens in normal_tokens]
        remove_punc_in_tokens = [[self.bdp.remove_punc_in_token(token) for token in tokens] for tokens in remove_punc_tokens]
        ners = self.bdp.ner_tagging(remove_punc_in_tokens)
        return ners, remove_punc_tokens

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
        if 'rank' in raw_q_sent:
            return ['ORDINAL'], 'rank'
        elif 'average' in raw_q_sent:
            return ['NUMBER', 'MONEY'], 'average'
        elif wh == 'what':
            if 'what century' in raw_q_sent:
                return ['ORDINAL'], 'century'
            if 'what language' in raw_q_sent:
                return ['NATIONALITY'], 'language'
            if 'nationality' in raw_q_sent:
                return ['NATIONALITY', 'PERSON'], 'nationality'
            if 'length' in raw_q_sent:
                return ['NUMBER'], 'length'
            if 'what year' in raw_q_sent:
                return ['DATE'], 'year'
            if 'what date' in raw_q_sent:
                return ['DATE'], 'date'
            if 'what percent' in raw_q_sent or 'what percentage' in raw_q_sent:
                return ['PERCENT'], 'percentage'
            if 'number' in raw_q_sent:
                return ['NUMBER'], 'number'
            if 'in what place' in raw_q_sent:
                return ['ORDINAL'], 'order'
            if 'what country' in raw_q_sent:
                return ['COUNTRY'], 'country'
            if 'what city' in raw_q_sent:
                return ['STATE_OR_PROVINCE', 'CITY', 'LOCATION'], 'city'
            if 'what region' in raw_q_sent:
                return ['NATIONALITY'], 'region'
            if 'location' in raw_q_sent:
                return ['LOCATION'], 'place'
            if 'population' in raw_q_sent:
                return ['PERCENT', 'NUMBER'], 'population'
            if 'fraction' in raw_q_sent:
                return ['ORDINAL'], 'fraction'
            if 'what age' in raw_q_sent:
                return ['NUMBER'], 'age'
            if 'what decade' in raw_q_sent:
                return ['DATE'], 'decade'
            if 'temperature' in raw_q_sent:
                return ['NUMBER'], 'temperature'
            if 'abundance' in raw_q_sent:
                return ['PERCENT'], 'abundance'
            if 'capacity' in raw_q_sent:
                return ['NUMBER'], 'capacity'
            else:
                return ['O', 'OTHER', 'PERSON', 'LOCATION', 'NUMBER'], 'else'
        elif wh == 'when':
            return ['DATE', 'TIME', 'NUMBER'], 'time'
        elif wh == 'who' or wh == 'whom':
            return ['PERSON', 'ORGANIZATION', 'OTHER'], 'person'
        elif wh == 'where':
            return ['LOCATION', 'ORDINAL', 'OTHER'], 'location'
        elif wh == 'how':
            if 'old' in raw_q_sent or 'large' in raw_q_sent:
                return ['NUMBER'], 'number'
            elif 'how long' in raw_q_sent:
                return ['DURATION', 'NUMBER'], 'length'
            elif 'how far' in raw_q_sent or 'how fast' in raw_q_sent:
                return ['NUMBER', 'TIME', 'PERCENT'], 'length'
            elif 'how many' in raw_q_sent:
                return ['NUMBER'], 'times'
            elif 'how much money' in raw_q_sent:
                return ['MONEY', 'PERCENT', 'NUMBER'], 'money'
            elif 'how much' in raw_q_sent:
                return ['MONEY', 'PERCENT', 'NUMBER'], 'money'
            elif 'how tall' in raw_q_sent:
                return ['number'], 'tall'
            else:
                return ['O', 'NUMBER', 'LOCATION', 'PERSON', 'ORGANIZATION'], 'else'
        elif wh == 'which':
            if 'which language' in raw_q_sent:
                return ['NATIONALITY'], 'language'
            if 'which year' in raw_q_sent:
                return ['TIME', 'NUMBER'], 'year'
            if 'place' in raw_q_sent or 'country' in raw_q_sent or 'city' in raw_q_sent or 'location' in raw_q_sent or 'site' in raw_q_sent:
                return ['LOCATION', 'ORGANIZATION', 'OTHER', 'PERSON'], 'place'
            if 'person' in raw_q_sent:
                return ['PERSON', 'ORGANIZATION', 'OTHER', 'LOCATION'], 'person'
            else:
                return ['O', 'OTHER', 'LOCATION', 'PERSON', 'NUMBER'], 'else'
        elif 'activism' in raw_q_sent or 'philosophy' in raw_q_sent or 'ideology' in raw_q_sent:
            return ['IDEOLOGY'], 'ideology'
        elif 'war' in raw_q_sent or 'blood' in raw_q_sent:
            return ['CAUSE_OF_DEATH'], 'war'
        else:
            return ['O', 'OTHER', 'LOCATION', 'PERSON', 'NUMBER'], 'else'

    def pred_answer_type(self, par_id, entities, qs_processed,
                                         possible_qs_type_rank, qs_type):


        # not_in_qs_entities = self.remove_entity_in_qs(qs_processed, entities)

        # qs_entities = [self.bdp.lemmatize(item[0]) for item in qs_entities]
        # not_in_qs_entities = self.remove_entity_in_qs(qs_entities, entities)
        #doubt!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        not_in_qs_entities = self.remove_entity_in_qs(qs_processed, entities)
        ner_type_to_entities_dict = self.get_ner_type_to_entities_dict(not_in_qs_entities)
        grouped_entities_strings_lemmatized = [self.bdp.lemmatize_entity_name(tup[0]) for tup in entities]
        # print("ner_type_to_entities_dict",ner_type_to_entities_dict)
        if not possible_qs_type_rank:
            return -1, [], []

        for type in possible_qs_type_rank:
            if len(ner_type_to_entities_dict[type]) != 0:
                    assert ner_type_to_entities_dict[type]
                    one_type_entities = ner_type_to_entities_dict[type]
                    # print(one_type_entities)
                    one_type_grouped_entities_strings = [x[0] for x in one_type_entities]
                    # print(one_type_grouped_entities_strings)
                    # if type == 'O' or type == 'OTHER':
                    if type == 'O':
                        one_type_grouped_entities_strings = [x[0] for x in pos_tag(one_type_grouped_entities_strings)
                                                            if 'NN' in x[1]]
                    distance = []
                    possible_entity_pos = []
                    qs_token_in_entity_pos = []

                    # print("one_type_grouped_entities_strings", one_type_grouped_entities_strings)

                    for qs_token in qs_processed:
                        if qs_token in grouped_entities_strings_lemmatized:
                            for i in range(len(grouped_entities_strings_lemmatized)):
                                entity_string = grouped_entities_strings_lemmatized[i]
                                if entity_string.lower() in qs_token:
                                    qs_token_in_entity_pos.append(i)
                    for entity in one_type_grouped_entities_strings:
                        for j in range(len(entities)):
                            word = entities[j][0]
                            if word.lower() == entity.lower():
                                sum_dist = 0
                                for k in qs_token_in_entity_pos:
                                    sum_dist += (abs(j - k))
                                distance.append(sum_dist)
                                possible_entity_pos.append(j)
                                break
                    assert len(possible_entity_pos) == len(distance)

                    #             min_idx = np.argmin(distance)
                    #
                    #             best_entity = one_type_grouped_entities_strings[min_idx]
                    #             if qs_type == 'year' and type == 'NUMBER':
                    #                 years = [x for x in one_type_grouped_entities_strings if len(x) == 4]
                    #                 if len(best_entity) != 4 and years:
                    #                     best_entity = years[0]
                    #                     return best_entity.lower(), possible_qs_type_rank
                    #
                    #             chunk_entities = entities[:possible_entity_pos[min_idx]]
                    #             actual_idx = 0

                    if distance:
                        min_idx = np.argmin(distance)
                        best_entity = one_type_grouped_entities_strings[min_idx]
                        if qs_type == 'year':
                            while len(best_entity) != 4 and len(distance) > 1:
                                distance.remove(distance[min_idx])
                                min_idx = np.argmin(distance)
                                best_entity = one_type_grouped_entities_strings[min_idx]
                            return best_entity.lower(), possible_qs_type_rank, one_type_grouped_entities_strings
                        #             chunk_entities = entities[:possible_entity_pos[min_idx]]
                        #             actual_idx = 0
                        #             for entity in chunk_entities:
                        #                   actual_idx += len(entity[0].split())
                        return best_entity.lower(), possible_qs_type_rank, one_type_grouped_entities_strings
        return -1, [], []

                        # if qs_type == 'year' and type == 'NUMBER':
                        #     years = [x for x in one_type_grouped_entities_strings if len(x) == 4]
                        #     if len(best_entity) != 4 and years:
                        #         best_entity = years[0]
                        #         return best_entity.lower(), possible_qs_type_rank

                        # chunk_entities = entities[:possible_entity_pos[min_idx]]
                        # actual_idx = 0
                        # for entity in chunk_entities:
                        #       actual_idx += len(entity[0].split())
                        # actual_end_idx = actual_idx + len(best_entity.split())
                        # if qs_type == 'length' and actual_end_idx < len(original_par):
                        #     actual_end_idx += 1

        # original_par = ner_par.copy()
        # entities = self.get_combined_entities(ner_par)
        # assert len(original_par) == len(ner_par)
        # not_in_qs_entities = self.remove_entity_in_qs(qs_processed, entities)
        # qs_processed = self.bdp.remove_stop_words(qs_processed)
        # ner_type_to_entities_dict = self.get_ner_type_to_entities_dict(not_in_qs_entities)
        # grouped_entities_strings_lemmatized = [self.bdp.lemmatize(tup[0]) for tup in entities]
        # for type in possible_qs_type_rank:
        #     if len(ner_type_to_entities_dict[type]) != 0:
        #         assert ner_type_to_entities_dict[type]
        #         one_type_entities = ner_type_to_entities_dict[type]
        #         # print(one_type_entities)
        #         one_type_grouped_entities_strings = [x[0] for x in one_type_entities]
        #         one_type_grouped_entities_strings = self.bdp.remove_stop_words(one_type_grouped_entities_strings)
        #         # print(one_type_grouped_entities_strings)
        #         if type == 'O':
        #             one_type_grouped_entities_strings = [x[0] for x in pos_tag(one_type_grouped_entities_strings)
        #                                                 if 'NN' in x[1]]
        #         distance = []
        #         possible_entity_pos = []
        #
        #         qs_token_in_entity_pos = []
        #         # print(qs_processed)
        #         #
        #         # for qs_token in qs_processed:
        #         #     for i in range(len(grouped_entities_strings_lemmatized)):
        #         #         entity_string = grouped_entities_strings_lemmatized[i]
        #         #         if qs_token in entity_string:
        #         #             qs_token_in_entity_pos.append(i)
        #
        #         for qs_token in qs_processed:
        #             if qs_token in grouped_entities_strings_lemmatized:
        #                 for i in range(len(grouped_entities_strings_lemmatized)):
        #                     entity_string = grouped_entities_strings_lemmatized[i]
        #                     if entity_string.lower() == qs_token.lower():
        #                         qs_token_in_entity_pos.append(i)
        #         for entity in one_type_grouped_entities_strings:
        #             for j in range(len(entities)):
        #                 word = entities[j][0]
        #                 if word.lower() == entity.lower():
        #                     sum_dist = 0
        #                     for k in qs_token_in_entity_pos:
        #                         sum_dist += (abs(j - k))
        #                     distance.append(sum_dist)
        #                     possible_entity_pos.append(j)
        #                     break
        #         assert len(possible_entity_pos) == len(distance)
        #         if distance:
        #             min_idx = np.argmin(distance)
        #
        #             best_entity = one_type_grouped_entities_strings[min_idx]
        #             if qs_type == 'year' and type == 'NUMBER':
        #                 years = [x for x in one_type_grouped_entities_strings if len(x) == 4]
        #                 if len(best_entity) != 4 and years:
        #                     best_entity = years[0]
        #                     return best_entity.lower(), possible_qs_type_rank
        #
        #             chunk_entities = entities[:possible_entity_pos[min_idx]]
        #             actual_idx = 0
        #             for entity in chunk_entities:
        #                   actual_idx += len(entity[0].split())
        #             actual_end_idx = actual_idx + len(best_entity.split())
        #             if qs_type == 'length' and actual_end_idx < len(original_par):
        #                 actual_end_idx += 1
        #             # if type=='NUMBER' and qtype=='length':
        #             #     if actual_end_ind < len(raw_doc):
        #             #         next_word = raw_doc[actual_end_ind]
        #             #         synsets = wordnet.synsets(next_word)
        #             #         if synsets:
        #             #             best_syn = synsets[0]
        #             #             hypers = best_syn.hypernyms()
        #             #             if hypers:
        #             #                 best_hyper = hypers[0]
        #             #                 lemmas = best_hyper.lemma_names()
        #             #                 if lemmas:
        #             #                     lemma = lemmas[0]
        #             #                     if 'unit' in lemma:
        #             #                         actual_end_ind += 1
        #             #
        #
        #             actual_best_entity = ' '.join(original_par[actual_idx:actual_end_idx])
        #             actual_best_entity = actual_best_entity.replace('\"', '')
        #             actual_best_entity = actual_best_entity.replace(',', '-COMMA-')
        #             if qs_type == 'percentage' and 'per' not in actual_best_entity and actual_best_entity[-1].isdigit():
        #                 if actual_end_idx < len(original_par):
        #                     if original_par[actual_end_idx] == 'per':
        #                         actual_best_entity += ' per cent'
        #                     elif original_par[actual_end_idx] == 'percent':
        #                         actual_best_entity += ' percent'
        #                     else:
        #                         actual_best_entity += ' %'
        #                 else:
        #                     actual_best_entity += ' %'
        #             if qs_type == 'money' and actual_best_entity[0].isdigit():
        #                 actual_best_entity = '$' + actual_best_entity
        #
        #             return actual_best_entity.lower(), possible_qs_type_rank
        # return -1, []

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
            entity_item = [' '.join(entity), ner_type]
            entities.append(entity_item)
        return entities

    def remove_entity_in_qs(self, qs, entities):
        valid_entities = []
        for entity in entities:
            entity_words = entity[0].split()
            for word in entity_words:
                word = word.lower()
                if self.bdp.lemmatize(word) not in qs:
                    valid_entities.append(entity)
                    break
        return valid_entities

    def get_ner_type_to_entities_dict(self, entities):
        ner_type_to_entities_dict = defaultdict(list)
        for entity in entities:
            ner_type = entity[1]
            ner_type_to_entities_dict[ner_type].append(entity)
        return ner_type_to_entities_dict

    def preprocess_questions(self, raw_qs):
        raw_split = word_tokenize(raw_qs.replace("\u200b", '').replace("\u2014", ''))
        remove_pure_punc = [token for token in raw_split if not self.bdp.is_pure_puncs(token)]
        remove_punc_in_words = [self.bdp.remove_punc_in_token(token) for token in remove_pure_punc]
        lemmatized = self.bdp.lemmatize_tokens(remove_punc_in_words)
        return lemmatized

    def ner_process(self, text):
        ner_par = self.bdp.nlp.ner(text)
        original_ner = []
        for tup in ner_par:
            tup = list(tup)
            if tup[1] in self.other:
                tup[1] = 'OTHER'
            tup[0] = self.bdp.remove_punc_in_token_for_rule(tup[0])
            original_ner.append(tup)
        original_ner = self.get_combined_entities(original_ner)
        original_ner = [item for item in original_ner if not self.bdp.is_pure_puncs(item[0])]
        original_ner = [item for item in original_ner if item[0].lower() not in stopwords.words("english")]
        # original_ner = [item for item in original_ner if not self.entity_lenth_filter(item)]
        return original_ner

    def find_pos_in_entities(self, entities, name):
        for i in range(len(entities)):
            word = self.bdp.lemmatize(entities[i][0])
            if word == name.lower():
                return i
        return -1

    def entity_lenth_filter(self, entity):
        if entity[1] == 'O' and len(entity[0]) <= 2:
            return True
        else:
            return False

if __name__ == '__main__':
    rule_based_QA = RuleBasedQA()