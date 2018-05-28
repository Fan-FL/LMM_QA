from nltk import word_tokenize
from nltk import sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag, pos_tag_sents
from data import Data
from config import Config
import pickle
import numpy as np
from scnn import Trainer


class BasicDataProcessor:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path,
                                                                binary=True)
        self.trn = Trainer()
        # self.word2vec_model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path)

    def train_sens_embeddings(self):
        question_vectors = []
        sent_vectors = []
        label = []
        total = int(len(self.data.train_qs_processed))
        for i in range(total):
            # for i in range(20):
            if i % 1000 == 0:
                print(i, '/', total)
            train_qs = self.data.train_qs_processed[i]
            train_answer = self.data.train_answers[i]
            train_par_id = self.data.train_answer_par_ids[i]
            train_doc_id = self.data.train_doc_ids[i]
            train_par = self.data.doc_texts[train_doc_id][train_par_id]
            train_sents = sent_tokenize(train_par)
            q_vector = []
            for token in train_qs:
                if token in self.word2vec_model.vocab:
                    q_vector.append(list(self.word2vec_model[token]))

            if len(q_vector) > 30:
                continue

            if len(q_vector) < 30:
                for k in range(30 - len(q_vector)):
                    q_vector.append([0] * 300)

            for sent_id in range(len(train_sents)):
                if train_answer in train_sents[sent_id]:
                    sent_vector = self.generate_sent_embedding(train_sents[sent_id])
                    if sent_vector:
                        question_vectors.append(q_vector)
                        sent_vectors.append(sent_vector)
                        label.append(1)

                        for wrong_id in range(len(train_sents)):
                            if wrong_id == sent_id:
                                continue
                            else:
                                wrong_sent_vector = self.generate_sent_embedding(
                                    train_sents[wrong_id])
                                if wrong_sent_vector:
                                    question_vectors.append(q_vector)
                                    sent_vectors.append(wrong_sent_vector)
                                    label.append(0)
                        break
                        # if sent_id > 0:
                        #     wrong_id = sent_id - 1
                        #     wrong_sent_vector = self.generate_sent_embedding(train_sents[wrong_id])
                        #     if wrong_sent_vector:
                        #         question_vectors.append(q_vector)
                        #         sent_vectors.append(wrong_sent_vector)
                        #         label.append(0)
                        # elif sent_id + 1 < len(train_sents):
                        #     wrong_id = sent_id + 1
                        #     wrong_sent_vector = self.generate_sent_embedding(train_sents[wrong_id])
                        #     if wrong_sent_vector:
                        #         question_vectors.append(q_vector)
                        #         sent_vectors.append(wrong_sent_vector)
                        #         label.append(0)

        # aaa = np.array(question_vectors)
        # bbb = np.array(sent_vectors)
        # ccc = np.array(label)
        # print(question_vectors)
        # print(sent_vectors)
        # print(label)
        # print(aaa.shape, bbb.shape, ccc.shape)
        # print(sum(ccc))
        # for i in range(len(bbb)):
        #     if np.array(bbb[i]).shape != (30,300):
        #         print(np.array(bbb[i]).shape)
        #         print(ccc[i])

        self.trn.load_data(question_vectors, sent_vectors, label)
        self.trn.run()
        # with open(self.config.sentence_embedding_pkl, 'wb') as f:
        #     pickle.dump([question_vectors, sent_vectors, label], f)

    def generate_sent_embedding(self, sent):
        sent_vector = []
        sent_tokens = self.process_sent(sent)
        for sent_token in sent_tokens:
            if sent_token in self.word2vec_model.vocab:
                sent_vector.append(list(self.word2vec_model[sent_token]))
        if len(sent_vector) > 30:
            return []
        elif len(sent_vector) < 30:
            for k in range(30 - len(sent_vector)):
                sent_vector.append([0] * 300)
        return sent_vector

    def preprocess_question(self, question):
        normal_tokens = word_tokenize(question.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        remove_punc_in_tokens = [self.remove_punc_in_token(token) for token in remove_punc_tokens]
        lower_tokens = self.lower_tokens(remove_punc_in_tokens)
        lemmatized_tokens = self.lemmatize_tokens(lower_tokens)
        return lemmatized_tokens

    def is_pure_puncs(self, tokens):
        if all([token in punctuation for token in tokens]):
            return True
        return False

    def remove_punc_in_token(self, token):
        return ''.join([x for x in token if x not in punctuation]).strip()

    def remove_stop_words(self, words):
        return [word for word in words if word.lower() not in stopwords.words("english")]

    def lemmatize_tokens(self, words):
        return [self.lemmatize(word) for word in words]

    def lemmatize(self, word):
        word = word.lower()
        lemma = self.lemmatizer.lemmatize(word, 'v')
        if lemma == word:
            lemma = self.lemmatizer.lemmatize(word, 'n')
        return lemma

    # def preprocess_doc(self, doc):
    #     normal_tokens = [[word_tokenize(sent.replace(u"\u200b",'').replace(u"\u2014",'')) for sent in sent_tokenize(par)] for par in doc]
    #     remove_punc_tokens = [[[token for token in tokens if not self.is_pure_puncs(token)] for tokens in par] for par in normal_tokens]
    #     remove_punc_in_tokens = [[[self.remove_punc_in_token(token) for token in tokens] for tokens in par] for par in remove_punc_tokens]
    #     ner_tags = [self.ner_tagging(par) for par in remove_punc_in_tokens]
    #     replaced_tokens = [
    #         [['number' if tup[1] == 'NUMBER' else 'person' if tup[1] == 'PERSON' else 'location' if tup[1] == 'LOCATION'
    #         else tup[0].lower() for tup in sent] for sent in par] for par in ner_tags]
    #     original_tokens = [[[tup[0] for tup in sent] for sent in par] for par in ner_tags]
    #     remove_stop_tokens_replaced = [[self.remove_stop_words(tokens) for tokens in par] for par in replaced_tokens]
    #     lemmatized_tokens_replaced = [[self.lemmatize_tokens(tokens) for tokens in par] for par in remove_stop_tokens_replaced]
    #     doc_par_tokens = []
    #     for par in lemmatized_tokens_replaced:
    #         par_tokens = []
    #         for sens in par:
    #             par_tokens += sens
    #         doc_par_tokens.append(par_tokens)
    #     return lemmatized_tokens_replaced, original_tokens, ner_tags, doc_par_tokens

    def preprocess_doc(self, doc):
        normal_tokens = [word_tokenize(par.replace(u"\u200b", '').replace(u"\u2014", '')) for par in
                         doc]
        remove_punc_tokens = [[token for token in tokens if not self.is_pure_puncs(token)] for
                              tokens in normal_tokens]
        remove_punc_in_tokens = [[self.remove_punc_in_token(token) for token in tokens] for tokens
                                 in remove_punc_tokens]
        lower_tokens = [self.lower_tokens(tokens) for tokens in remove_punc_in_tokens]
        lemmatized_tokens = [self.lemmatize_tokens(tokens) for tokens in lower_tokens]
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def process_sent(self, sens):
        normal_tokens = word_tokenize(sens.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        replace_numbers = []
        for token in remove_punc_tokens:
            if any([x.isdigit() for x in token]):
                replace_numbers.append('number')
            else:
                replace_numbers.append(token)
        lower_tokens = self.lower_tokens(replace_numbers)
        lemmatized_tokens = self.lemmatize_tokens(lower_tokens)
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def ner_tagging(self, sents):
        ner_sents = self.tagger.tag_sents(sents)
        processed_ners = []
        for i in range(len(ner_sents)):
            sent = sents[i]
            pos_sent = pos_tag(sent)
            ner_sent = ner_sents[i]
            processed_ner_sent = []
            for j in range(len(ner_sent)):
                span, tag = ner_sent[j]
                _, pos = pos_sent[j]
                if span.isdigit() or pos == 'CD':
                    processed_ner_sent.append((span, 'NUMBER'))
                # elif tag == 'PERSON':
                #     processed_ner_sent.append((span, 'PERSON'))
                # elif tag == 'LOCATION':
                #     processed_ner_sent.append((span, 'LOCATION'))
                elif tag == 'ORGANIZATION':
                    processed_ner_sent.append((span, 'OTHER'))
                elif j != 0 and tag == 'O' and span[0].isupper():
                    processed_ner_sent.append((span, 'OTHER'))
                else:
                    processed_ner_sent.append((span, tag))
            processed_ners.append(processed_ner_sent)
        return processed_ners


if __name__ == '__main__':
    data = Data()
    config = Config()
    bdp = BasicDataProcessor(config, data)
