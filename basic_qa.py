# -*- coding: utf-8 -*-

from data_processor import DataProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import math
import numpy as np
import unicodecsv as csv
from rb_classifier import RuleBasedClassifier
import sys
from nltk import pos_tag
from nltk.corpus import wordnet

class Basic_QA:
    def __init__(self, enhance=None):

        self.dataProcessor = DataProcessor()
        self.config = self.dataProcessor.config
        self.ruleBasedClassifier = RuleBasedClassifier(self.config)
        self.load_raw_doc()
        self.load_raw_dev()
        self.test_on_all_dev_qs()
        if enhance:
            self.dataProcessor.load_word2vec_model()
            self.dataProcessor.load_qtype_model()

    def load_raw_training(self):
        self.docids, self.questions, self.answers, self.answer_paragraphs, _ = self.dataProcessor.load_raw_train()

    def load_raw_doc(self):
        self.doc_docids, _, self.texts, _, _ = self.dataProcessor.load_raw_doc()

    def load_raw_dev(self):
        self.dev_docids, self.dev_questions, self.dev_answers, self.dev_answer_paragraphs, _ = self.dataProcessor.load_raw_dev()

    def load_raw_test(self):
        self.test_docids, self.test_questions, _, _, self.test_ids = self.dataProcessor.load_raw_test()

    def test_on_all_test_qs(self):
        with open('test_results.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['id', 'answer'])
            for i in range(len(self.test_ids)):
                print(i, " / ", len(self.test_ids))
                test_id = self.test_ids[i]
                test_qs = self.test_questions[i]
                test_docid = self.test_docids[i]
                test_doc = self.texts[test_docid]

                processed_test_qs = self.dataProcessor.preprocess_questions(test_qs)
                split_test_raw_doc, ner_test_doc, processed_test_doc = self.dataProcessor.preprocess_doc(test_doc)
                wh = self.ruleBasedClassifier.extract_wh_word(processed_test_qs)
                type_rank, q_type = self.ruleBasedClassifier.simple_rules(wh, processed_test_qs)
                answer, types, _ = self.process_query_on_doc_bm25(
                    processed_test_qs, processed_test_doc, ner_test_doc, split_test_raw_doc, type_rank, q_type)
                csv_writer.writerow([test_id, answer])


    def enhance_test_on_all_test_qs(self, save=True):
        total = 0
        classes = self.dataProcessor.qtype_model.classes_
        with open('test_enhance.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['id', 'answer'])
            for i in range(len(self.test_wikis)):
                print(i)
                test_wiki = self.test_wikis[i]
                test_qs = self.test_questions[i]
                processed_test_qs = self.dataProcessor.preprocess_questions(test_qs)
                dep_qs = self.dataProcessor.dep_parse_sents(processed_test_qs)
                split_raw_wiki, ner_test_wiki, processed_test_wiki = self.dataProcessor.preprocess_wiki(test_wiki)
                for j in range(len(processed_test_qs)):
                    raw_q = test_qs[j]
                    q = processed_test_qs[j]
                    wh = self.ruleBasedClassifier.extract_wh_word(q)
                    type_rank, q_type = self.ruleBasedClassifier.simple_rules(wh, q)
                    if q_type == 'else':
                        dep_q = dep_qs[j]
                        q_vec = self.dataProcessor.q_to_vec(q, dep_q)
                        probs = self.dataProcessor.qtype_model.predict_proba([q_vec])[0]
                        p = list(zip(classes, probs))
                        p = sorted(p, key=lambda x: x[1], reverse=True)
                        type_rank = [x[0] for x in p]
                        print(type_rank)
                    answer, types, predicted_ind = self.process_query_on_wiki_bm25(q, processed_test_wiki, ner_test_wiki, split_raw_wiki, type_rank, q_type)
                    csv_writer.writerow([total, answer])
                total += 1



    def test_on_all_dev_qs(self, save=True):
        correct = 0
        correct_id = 0
        total = len(self.dev_questions)
        # total = len(self.dev_questions)
        with open('dev.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['W/R', 'query', 'predicted_id', 'actual_id', 'predicted_answer', 'actual_answer', 'predicted_answer_type'])
            # for i in range(len(self.dev_questions)):
            for i in range(3,5):
                # print(i, " / ", len(self.dev_questions))

                dev_qs = self.dev_questions[i]
                doc_id = self.dev_docids[i]
                dev_doc = self.texts[doc_id]
                dev_answer = self.dev_answers[i]
                dev_answer_par_id = self.dev_answer_paragraphs[i]
                processed_dev_qs = self.dataProcessor.preprocess_questions(dev_qs)
                split_dev_raw_doc, ner_dev_doc, processed_dev_doc = self.dataProcessor.preprocess_doc(dev_doc)

                wh = self.ruleBasedClassifier.extract_wh_word(processed_dev_qs)
                type_rank, q_type = self.ruleBasedClassifier.simple_rules(wh, processed_dev_qs)
                # print type_rank, q_type
                answer, types, predicted_id = self.process_query_on_doc_bm25(
                    processed_dev_qs, processed_dev_doc, ner_dev_doc, split_dev_raw_doc, type_rank, q_type)
                types = ' '.join(types)
                if predicted_id == int(dev_answer_par_id):
                    correct_id += 1
                if dev_answer == answer:
                    csv_writer.writerow(["##right##", dev_qs, predicted_id, dev_answer_par_id, answer, dev_answer, types])
                    correct += 1
                else:
                    csv_writer.writerow(["##wrong##", dev_qs, predicted_id, dev_answer_par_id, answer,  dev_answer, types])
                # print(dev_answer, " ; ", answer)
                # print "correct :", correct
            csv_writer.writerow([str(correct), str(correct * 100.0 / total)])
            csv_writer.writerow([str(correct_id), str(correct_id * 100.0 / total)])
            csv_writer.writerow([str(total)])
        print(correct*100.0/total)
        print(correct_id*100.0/total)


    def enhance_test_on_all_dev_qs(self, save=True):
        correct = 0
        correct_ind = 0
        total = 0
        classes = self.dataProcessor.qtype_model.classes_
        with open('dev_enhance.csv', 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['query', 'predicted_ind', 'predicted_answer', 'actual_ind', 'actual_answer', 'predicted_answer_type'])
            for i in range(len(self.dev_wikis)):
                print(i)
                dev_wiki = self.dev_wikis[i]
                dev_qs = self.dev_questions[i]
                dev_answer_inds = self.dev_answer_inds[i]
                dev_answers = self.dev_answers[i]
                processed_dev_qs = self.dataProcessor.preprocess_questions(dev_qs)
                dep_qs = self.dataProcessor.dep_parse_sents(processed_dev_qs)
                split_raw_wiki, ner_dev_wiki, processed_dev_wiki = self.dataProcessor.preprocess_wiki(dev_wiki)
                for j in range(len(processed_dev_qs)):
                    print(total)
                    raw_q = dev_qs[j]
                    q = processed_dev_qs[j]
                    wh = self.ruleBasedClassifier.extract_wh_word(q)
                    type_rank, q_type = self.ruleBasedClassifier.simple_rules(wh, q)
                    if q_type == 'else':
                        dep_q = dep_qs[j]
                        q_vec = self.dataProcessor.q_to_vec(q, dep_q)
                        probs = self.dataProcessor.qtype_model.predict_proba([q_vec])[0]
                        p = list(zip(classes, probs))
                        prob_dict = defaultdict(float)
                        for c, prob in p:
                            prob_dict[c] = prob
                        if prob_dict['O'] > prob_dict['OTHER']:
                            tmp = prob_dict['O']
                            prob_dict['O'] = prob_dict['OTHER']
                            prob_dict['OTHER'] = tmp
                        p = list(prob_dict.items())
                        p = sorted(p, key=lambda x: x[1], reverse=True)
                        answer, types, predicted_ind = self.enhance_process_query_on_wiki_bm25(p, q, processed_dev_wiki, ner_dev_wiki, split_raw_wiki, q_type)
                    else:
                        answer, types, predicted_ind = self.process_query_on_wiki_bm25(q, processed_dev_wiki, ner_dev_wiki, split_raw_wiki, type_rank, q_type)
                    actual_answer = dev_answers[j]
                    actual_ind = dev_answer_inds[j]
                    types = ' '.join(types)

                    csv_writer.writerow([raw_q, predicted_ind, answer, actual_ind, actual_answer, types])
                    if actual_answer == answer:
                        correct += 1
                    total += 1
        print(correct*100.0/total)


    def enhance_process_query_on_wiki_bm25(self, type_probs, query, wiki, ner_wiki, split_raw_wiki, q_type):
        ranked_docs = self.dataProcessor.get_best_doc(query, wiki)
        if not ranked_docs:
            return 'not sure', [], -1
        doc_type_prob_dict = {}
        for ind, bm in ranked_docs:
            for type, prob in type_probs:
                doc_type_prob_dict[(ind, type)] = bm*prob
        doc_type_prob_pairs = sorted(list(doc_type_prob_dict.items()), key=lambda x:x[1], reverse=True)
        doc_type_prob_pairs = [x[0] for x in doc_type_prob_pairs]
        best_entity, types, doc_ind = self.rank_entities_from_answer_doc_prob_priority(doc_type_prob_pairs, query,
                                                                                           ner_wiki, wiki,
                                                                                           split_raw_wiki, q_type)

        return best_entity, types, doc_ind


    def process_query_on_doc_bm25(self, query, doc, ner_doc, split_raw_doc, type_rank, q_type):
        ranked_docs = self.dataProcessor.get_best_par(query, doc)
        if not ranked_docs:
            return 'not sure', [], -1
        ranked_par_ids = [x[0] for x in ranked_docs]
        best_entity, types, doc_ind = self.rank_entities_from_answer_doc_same_doc_priority(ranked_par_ids, query,
                                                                                           ner_doc, doc,
                                                                                           split_raw_doc, type_rank, q_type)

        return best_entity.lower(), types, doc_ind


    def rank_qa_sents(self, query, wiki_inv_dict):
        # print query
        doc_score_dict = defaultdict(float)
        for term in set(query):
            tfidf_dict = wiki_inv_dict[term]
            for doc, tfidf in list(tfidf_dict.items()):
                doc_score_dict[doc] += tfidf
        ranked_docs = sorted(list(doc_score_dict.items()), key=lambda x: x[1], reverse=True)
        if not ranked_docs:
            return -1
        # print ranked_docs
        return ranked_docs


    def combine_entity(self, ner_sent):
        combined_sent = []
        group = []
        prev_tag = ''
        for tup in ner_sent:
            tag = tup[1]
            if not prev_tag:
               group.append(tup)
               prev_tag = tag
            else:
                if tag == prev_tag:
                    group.append(tup)
                else:
                    combined_sent.append(group)
                    group = [tup]
                    prev_tag = tag
        combined_sent.append(group)
        return combined_sent

    def filter_existing_content(self, query, entities):
        valids = []
        filtered = []
        for entity in entities:
            entity_words = entity[0].split()
            is_valid = False
            for word in entity_words:
                word = word.lower()
                if self.dataProcessor.lemmatize(word) not in query:
                    valids.append(entity)
                    is_valid = True
                    break
            if not is_valid:
                filtered.append(entity)
        assert len(valids) + len(filtered) == len(entities)
        return valids, filtered

    def compute_entity_dict_by_type(self, entities):
        type_entity_dict = defaultdict(list)
        for entity in entities:
            entity_type = entity[1]
            type_entity_dict[entity_type].append(entity)
        return type_entity_dict

    def rank_entites_by_open_class_dist(self, type, type_entity_dict, query, entities, groups, raw_doc, qtype=None):
        #问题中所有有意义的词
        print(raw_doc)
        open_class_query = self.dataProcessor.remove_stop_words(query)
        doc_words_entities = [self.dataProcessor.lemmatize(tup[0]) for tup in entities]
        # print(doc_words_entities)
        assert type_entity_dict[type]
        # print "type_entity_dict :", type_entity_dict
        open_word_ind = []
        type_entities = type_entity_dict[type]
        # print "type_entities :", type_entities
        raw_type_entity_spans = [x[0] for x in type_entities]
        #所有想要词性的答案中的词
        raw_type_entity_spans = self.dataProcessor.remove_stop_words(raw_type_entity_spans)

        # print(raw_type_entity_spans)
        if type == 'O':
            raw_type_entity_spans = [x[0] for x in pos_tag(raw_type_entity_spans) if 'NN' in x[1]]
        dists = []
        js = []

        # print(open_class_query)
        for open_class_word in open_class_query:
            # if open_word_ind:
            #     break
            if open_class_word in doc_words_entities:
                for i in range(len(doc_words_entities)):
                    word = doc_words_entities[i]
                    if word.lower() == open_class_word.lower():
                        open_word_ind.append(i)
        for entity in raw_type_entity_spans:
            for j in range(len(entities)):
                word = entities[j][0]
                if word.lower() == entity.lower():
                    sum_dist = 0
                    for k in open_word_ind:
                        sum_dist += (abs(j-k))
                    dists.append(sum_dist)
                    js.append(j)
                    break
        assert len(js) == len(dists)
        if dists:
            min_dist_ind = np.argmin(dists)
            best_entity = raw_type_entity_spans[min_dist_ind]
            if qtype == 'year' and type == 'NUMBER':
                years = [x for x in raw_type_entity_spans if len(x) == 4]
                if len(best_entity) != 4 and years:
                    best_entity = years[0]
                    return best_entity
            best_entity_ind = js[min_dist_ind]
            chunk_entities = entities[:best_entity_ind]
            actual_ind = 0
            for entity in chunk_entities:
                actual_ind += len(entity[0].split())
            actual_end_ind = actual_ind + len(best_entity.split())
            if qtype == 'length' and actual_end_ind < len(raw_doc):
                actual_end_ind += 1
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

            actual_best_entity = ' '.join(raw_doc[actual_ind:actual_end_ind])
            actual_best_entity = actual_best_entity.replace('\"', '')
            actual_best_entity = actual_best_entity.replace(',', '-COMMA-')
            if qtype == 'percentage' and 'per' not in actual_best_entity and actual_best_entity[-1].isdigit():
                if actual_end_ind < len(raw_doc):
                    if raw_doc[actual_end_ind].lower() == 'per':
                        actual_best_entity += ' per cent'
                    elif raw_doc[actual_end_ind].lower() == 'percent':
                        actual_best_entity += ' percent'
                    else:
                        actual_best_entity += ' %'
                else:
                    actual_best_entity += ' %'
            if qtype == 'money' and actual_best_entity[0].isdigit():
                actual_best_entity = '$'+actual_best_entity

            return actual_best_entity
        return -1

    def process_groups_into_entites(self, groups):
        entities = []
        for group in groups:
            entity_tag = group[0][1]
            if entity_tag == 'O':
                for entity in group:
                    entities.append(entity)
            else:
                entity_span = []
                for entity in group:
                    entity_span.append(entity[0])
                whole_entity = (' '.join(entity_span), entity_tag)
                entities.append(whole_entity)
        return entities


    def rank_entities_from_answer_doc_prob_priority(self, ranked_doc_type_probs, query, ner_docs, docs, split_raw_docs, qtype):
        entities_dict = {}
        typedict_dict = {}

        for doc_ind, type in ranked_doc_type_probs:
            ner_doc = ner_docs[doc_ind]
            raw_doc = split_raw_docs[doc_ind]
            if doc_ind not in entities_dict:
                groups = self.combine_entity(ner_doc)
                entities = self.process_groups_into_entites(groups)
                entities_dict[doc_ind] = entities
                assert len(raw_doc) == len(ner_doc)
                candidates_phase1, filtered_phase1 = self.filter_existing_content(query, entities)
                candidates_phase2_dict = self.compute_entity_dict_by_type(candidates_phase1)
                typedict_dict[doc_ind] = candidates_phase2_dict
            else:
                entities = entities_dict[doc_ind]
                candidates_phase2_dict = typedict_dict[doc_ind]
            if len(candidates_phase2_dict[type]) != 0:
                best_entity = self.rank_entites_by_open_class_dist(type, candidates_phase2_dict, query,
                                                                   entities,
                                                                   [], raw_doc, qtype)
                if best_entity != -1:
                    return best_entity, [], doc_ind
        return 'not sure', [], -1

    def rank_entities_from_answer_doc_same_doc_priority(self, ranked_par_ids, query, ner_doc, doc, split_raw_doc, type_rank, q_type):
        for par_id in ranked_par_ids:
            ner_par = ner_doc[par_id]
            raw_par = split_raw_doc[par_id]
            groups = self.combine_entity(ner_par)
            entities = self.process_groups_into_entites(groups)
            assert len(raw_par) == len(ner_par)
            candidates_phase1, filtered_phase1 = self.filter_existing_content(
                self.dataProcessor.lower_sent(query), entities)
            # print(candidates_phase1)
            for type in type_rank:

                candidates_phase2_dict = self.compute_entity_dict_by_type(candidates_phase1)
                if len(candidates_phase2_dict[type]) != 0:
                    # print candidates_phase2_dict
                    best_entity = self.rank_entites_by_open_class_dist(type, candidates_phase2_dict, query,
                                                                       entities,
                                                                       groups, raw_par, q_type)
                    if best_entity != -1:
                        return best_entity, type_rank, ranked_par_ids[0]
        return 'not sure', [], ranked_par_ids[0]

if __name__ == '__main__':
    bQA = Basic_QA(enhance=0)

    # bQA.load_raw_training()

    bQA.load_raw_dev()
    bQA.test_on_all_dev_qs()
    # bQA.enhance_test_on_all_dev_qs()

    # bQA.load_raw_test()
    # bQA.test_on_all_test_qs()

