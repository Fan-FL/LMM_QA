import nltk
from data_processor import DataProcessor
from nltk.corpus import wordnet
class RuleBasedClassifier:
    def __init__(self, config):
        self.config = config
        self.tags = self.config.TAG
        self.whs = self.config.WH_words
        self.dataProcessor = DataProcessor()
        self.months = ['']

    def extract_wh_word(self, words):
        for word in words:
            if word.lower() in self.whs or word.lower() == 'whom':
                return word
        return -1

    def simple_rules(self, wh, q_words):
        lower = self.dataProcessor.lower_sent(q_words)
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


if __name__ == '__main__':
    rb = RuleBasedClassifier()