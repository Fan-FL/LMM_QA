from nltk import word_tokenize
from nltk import sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
from nltk.parse.corenlp import CoreNLPParser
from nltk import pos_tag, pos_tag_sents
from data import Data
from config import Config
from nltk.parse.stanford import StanfordDependencyParser
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
import nltk

class BasicDataProcessor:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.lemmatizer = WordNetLemmatizer()
        # self.word2vec_model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path, binary=False)
        # self.word2vec_model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path, binary=True)
        self.tagger = StanfordNERTagger(model_filename=self.config.ner_model_path)
        self.postagger = StanfordPOSTagger(path_to_jar=self.config.pos_jar_path, model_filename=self.config.pos_model_path)
        self.dependency_parser = StanfordDependencyParser(path_to_jar=self.config.parser_jar_path,
                                                          path_to_models_jar=self.config.parser_model_path)
        # self.nlp = StanfordCoreNLP("stanford/stanford-corenlp-full")
        self.nlp = StanfordCoreNLP("http://localhost", port=9000)

        self.punc = r"""!"#&'()*+;<=>?[]^`{}~"""

        # # text = ('5-Aug-25')
        # text = ('GOP Sen. Rand Paul was assaulted in his home in Bowling Green, Kentucky, on Friday, '
        #         'according to Kentucky State Police. State troopers responded to a call to the senator\'s '
        #         'residence at 3:21 p.m. Friday. Police arrested a man named Rene Albert Boucher, who they '
        #         'allege "intentionally assaulted" Paul, causing him "minor injury". Boucher, 59, of Bowling '
        #         'Green was charged with one count of fourth-degree assault. As of Saturday afternoon, he '
        #         'was being held in the Warren County Regional Jail on a $5,000 bond.')
        # aaa = self.nlp.parse(text)
        #
        # tree = nltk.tree.Tree.fromstring(aaa)
        # self.traverse_tree(tree)

    def traverse_tree(self, tree):
        # print("tree:", tree)
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                self.traverse_tree(subtree)
            print(type(subtree))






    def train_sens_embeddings(self):
        question_vectors = []
        sens_vectors = []
        label = []
        # for i in range(len(self.data.train_qs_processed)):
        for i in range(5):
            train_qs = self.data.train_qs_processed[i]
            train_answer = self.data.train_answers[i]
            train_par_id = self.data.train_answer_par_ids[i]
            train_doc_id = self.data.train_doc_ids[i]
            train_par = self.data.doc_texts[train_doc_id][train_par_id]
            train_sents = sent_tokenize(train_par)
            sens_vector = []
            q_vector = []
            for token in train_qs:
                if token in self.word2vec_model.vocab:
                    q_vector.append(list(self.word2vec_model[token]))
            for sent_id in range(len(train_sents)):
                if train_answer in train_sents[sent_id]:
                    sens_tokens = self.process_sent(train_sents[sent_id])
                    for sens_token in sens_tokens:
                        if sens_token in self.word2vec_model.vocab:
                            sens_vector.append(list(self.word2vec_model[sens_token]))
                    question_vectors.append(q_vector)
                    sens_vectors.append(sens_vector)
                    label.append(1)
        print(question_vectors)
        print(sens_vectors)
        print(label)

    def preprocess_questions(self, questions):
        return [self.preprocess_question(q) for q in questions]

    def process_docs(self, docs):
        return [self.preprocess_doc(doc) for doc in docs]

    # def process_docs(self, docs):
    #     doc_tokens_replaced = []
    #     doc_original_tokens = []
    #     doc_ner_tags = []
    #     doc_par_tokens_replaced = []
    #     i = 1
    #     # for doc in docs[:2]:
    #     for doc in docs:
    #         print(i, "/" , len(docs))
    #         i += 1
    #         tokens_replaced, original_tokens, ner_tags, par_tokens_replaced = self.preprocess_doc(doc)
    #         doc_tokens_replaced.append(tokens_replaced)
    #         doc_original_tokens.append(original_tokens)
    #         doc_ner_tags.append(ner_tags)
    #         doc_par_tokens_replaced.append(par_tokens_replaced)
    #     return doc_tokens_replaced, doc_original_tokens, doc_ner_tags, doc_par_tokens_replaced

    def preprocess_question(self, question):
        normal_tokens = word_tokenize(question.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        remove_punc_in_tokens = [self.remove_punc_in_token(token) for token in remove_punc_tokens]
        lower_tokens = self.lower_tokens(remove_punc_in_tokens)
        remove_stop_tokens = self.remove_stop_words(lower_tokens)
        for i in range(len(remove_stop_tokens)):
            if remove_stop_tokens[i] == 'where':
                remove_stop_tokens[i] = 'location'
            if remove_stop_tokens[i] == 'when':
                remove_stop_tokens[i] = 'time'
            if remove_stop_tokens[i] == 'who' or remove_stop_tokens[i] == 'whom':
                remove_stop_tokens[i] = 'person'
            if remove_stop_tokens[i] == 'why':
                remove_stop_tokens[i] = 'reason'
        lemmatized_tokens = self.lemmatize_tokens(remove_stop_tokens)
        return lemmatized_tokens

    def is_pure_puncs(self, token):
        if all([c in punctuation for c in token]):
            return True
        return False

    def remove_punc_in_token(self, token):
        return ''.join([x for x in token if x not in punctuation]).strip()

    def remove_punc_in_token_for_rule(self, token):
        return ''.join([x for x in token if x not in self.punc]).strip()

    def remove_stop_words(self, words):
        return [word for word in words if word.lower() not in stopwords.words("english")]

    def lemmatize_tokens(self, words):
        return [self.lemmatize(word.lower()) for word in words]

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
        normal_tokens = [word_tokenize(par.replace(u"\u200b",'').replace(u"\u2014",'')) for par in doc]
        remove_punc_tokens = [[token for token in tokens if not self.is_pure_puncs(token)] for tokens in normal_tokens]
        remove_punc_in_tokens = [[self.remove_punc_in_token(token) for token in tokens] for tokens in remove_punc_tokens]
        lower_tokens = [self.lower_tokens(tokens) for tokens in remove_punc_in_tokens]
        remove_stop_tokens = [self.remove_stop_words(tokens) for tokens in lower_tokens]
        lemmatized_tokens = [self.lemmatize_tokens(tokens) for tokens in remove_stop_tokens]
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def process_sent(self, sens):
        normal_tokens = word_tokenize(sens.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        remove_punc_in_tokens = [self.remove_punc_in_token(token) for token in remove_punc_tokens]
        ner_tags = self.sens_ner_tagging(remove_punc_in_tokens)
        replaced_tokens = ['number' if tup[1] == 'NUMBER' else 'person' if tup[1] == 'PERSON' else 'location' if tup[1] == 'LOCATION' else tup[0].lower() for tup in ner_tags]
        lower_tokens = self.lower_tokens(replaced_tokens)
        remove_stop_tokens = self.remove_stop_words(lower_tokens)
        lemmatized_tokens = self.lemmatize_tokens(remove_stop_tokens)
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

    def sens_ner_tagging(self, sent):
        ner_sents = self.tagger.tag_sents([sent])
        pos_sent = pos_tag(sent)
        ner_sent = ner_sents[0]
        processed_ner_sent = []
        for j in range(len(ner_sent)):
            span, tag = ner_sent[j]
            _, pos = pos_sent[j]
            if span.isdigit() or pos == 'CD':
                processed_ner_sent.append((span, 'NUMBER'))
            elif tag == 'PERSON':
                processed_ner_sent.append((span, 'PERSON'))
            elif tag == 'LOCATION':
                processed_ner_sent.append((span, 'LOCATION'))
            elif tag == 'ORGANIZATION':
                processed_ner_sent.append((span, 'OTHER'))
            elif j != 0 and tag == 'O' and span[0].isupper():
                processed_ner_sent.append((span, 'OTHER'))
            else:
                processed_ner_sent.append((span, tag))
        return processed_ner_sent

    def get_entity_names(self, entity):
        return [item[0].lower() for item in entity]

    def lemmatize_entity_name(self, entity_name):
        tokens = entity_name.split()
        tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)


if __name__ == '__main__':
    data = Data()
    config = Config()
    bdp = BasicDataProcessor(config, data)