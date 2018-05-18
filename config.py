class Config:
    def __init__(self):
        self.doc_fname = 'Data/documents.json'
        self.train_fname = 'Data/training.json'
        self.dev_fname = 'Data/devel.json'
        self.test_fname = 'Data/testing.json'

        self.embedding_size = 300
        self.word2vec_model_path = 'Google_word2vec/GoogleNews-vectors-negative300.bin'
        self.ner_model_path = 'stanford/stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz'
        self.ner_jar_path = 'stanford/stanford_ner/stanford-ner.jar'
        self.pos_model_path = 'stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'
        self.pos_jar_path = 'stanford/stanford-postagger/stanford-postagger.jar'
        self.parser_model_path = 'stanford/stanford-parser/stanford-parser-3.9.1-models.jar'
        self.parser_jar_path = 'stanford/stanford-parser/stanford-parser.jar'

        self.clf_save_path = './clf_withFeatures.pkl'



        self.WH_words = ['how', 'what', 'where', 'when', 'who', 'which']
        self.TAG = ['PERSON', 'LOCATION', 'NUMBER', 'OTHER', 'O']

        self.MAX_PAIR_NUM = 24
        self.q_type_clf_save_path = './models/clf_q.pkl'
        self.enhanced_ner_model_path = './stanford/classifiers/english.muc.7class.distsim.crf.ser.gz'
        self.enhanced_ner_TAG = ['ORGANIZATION', 'PERSON', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT',
                                 'FACILITY', 'GPE', 'OTHER']
        self.train_dev_processed_data_path = './data.pkl'
        self.answer_model_path = './answer_model.pkl'
