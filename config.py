class Config:
    def __init__(self):
        self.doc_file_name = 'Data/documents.json'
        self.train_file_name = 'Data/training.json'
        self.dev_file_name = 'Data/devel.json'
        self.test_file_name = 'Data/testing.json'

        self.embedding_size = 300
        # self.word2vec_model_path = 'model/pruned.word2vec.txt'
        self.word2vec_model_path = 'model/GoogleNews-vectors-negative300.bin'
        # self.ner_model_path = 'stanford/stanford_ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
        self.ner_model_path = 'stanford/stanford_ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
        self.ner_jar_path = 'stanford/stanford_ner/stanford-ner.jar'
        self.pos_model_path = 'stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'
        self.pos_jar_path = 'stanford/stanford-postagger/stanford-postagger.jar'
        self.parser_model_path = 'stanford/stanford-parser/stanford-parser-3.9.1-models.jar'
        self.parser_jar_path = 'stanford/stanford-parser/stanford-parser.jar'

        self.clf_save_path = './clf_withFeatures.pkl'
        self.doc_processed_path = 'pkl/doc_processed.pkl'
        self.train_qs_processed_path = 'pkl/train_qs_processed.pkl'
        self.dev_qs_processed_path = 'pkl/dev_qs_processed.pkl'
        self.test_qs_processed_path = 'pkl/test_qs_processed.pkl'
        self.dev_rule_pkl = 'pkl/dev_rule_pkl.pkl'
        self.sentence_embedding_pkl = 'pkl/sentence_embedding.pkl'
        self.training_ner_pkl = 'pkl/training_ner.pkl'
        self.dev_ner_pkl = 'pkl/dev_ner.pkl'

        self.doc_processed_path_bak = 'pkl_bak/doc_processed.pkl'
        self.train_qs_processed_path_bak = 'pkl_bak/train_qs_processed.pkl'
        self.dev_qs_processed_path_bak = 'pkl_bak/dev_qs_processed.pkl'
        self.test_qs_processed_path_bak = 'pkl_bak/test_qs_processed.pkl'
        self.dev_rule_pkl_bak = 'pkl_bak/dev_rule_pkl.pkl'

        self.WH_words = ['how', 'what', 'where', 'when', 'who', 'which']
        self.TAG = ['PERSON', 'LOCATION', 'NUMBER', 'OTHER', 'O']

        self.MAX_PAIR_NUM = 24
        self.q_type_clf_save_path = './models/clf_q.pkl'
        self.enhanced_ner_model_path = './stanford/classifiers/english.muc.7class.distsim.crf.ser.gz'
        self.enhanced_ner_TAG = ['ORGANIZATION', 'PERSON', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT',
                                 'FACILITY', 'GPE', 'OTHER']
        self.train_dev_processed_data_path = './data.pkl'
        self.answer_model_path = './answer_model.pkl'
