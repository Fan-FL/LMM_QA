from config import Config
import pickle
from file_loader import FileLoader
from data import Data
from basic_data_processor_sentence_embedding import BasicDataProcessor


class TrainBasedQA:
    def __init__(self):
        train = 0
        dev = 1
        test = 0
        load_processed_doc = 0
        load_doc_from_pkl = 0
        load_train_qs_from_pkl = 1
        load_dev_qs_from_pkl = 1
        load_test_qs_from_pkl = 1
        train_sens_embedding = 1
        dev_sens_embedding = 1

        self.data = Data()
        self.config = Config()
        self.fileLoader = FileLoader(self.config, self.data)
        self.bdp = BasicDataProcessor(self.config, self.data)

        self.fileLoader.load_doc()
        if load_processed_doc:
            if load_doc_from_pkl:
                with open(self.config.doc_processed_path, 'rb') as f:
                    self.data.doc_processed = pickle.load(f)
            else:
                self.data.doc_processed = self.bdp.process_docs(self.data.doc_texts)
                with open(self.config.doc_processed_path, 'wb') as f:
                    pickle.dump(self.data.doc_processed, f)

        if train:
            self.fileLoader.load_training_data()
            if load_train_qs_from_pkl:
                with open(self.config.train_qs_processed_path, 'rb') as f:
                    self.data.train_qs_processed = pickle.load(f)

            else:
                self.data.train_qs_processed = self.bdp.preprocess_questions(
                    self.data.train_questions)
                with open(self.config.train_qs_processed_path, 'wb') as f:
                    pickle.dump(self.data.train_qs_processed, f)

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

            if dev_sens_embedding:
                self.bdp.dev_sens_embeddings()

        if test:
            self.fileLoader.load_test_data()
            if load_test_qs_from_pkl:
                with open(self.config.test_qs_processed_path, 'rb') as f:
                    self.data.test_qs_processed = pickle.load(f)
            else:
                self.data.test_qs_processed = self.bdp.preprocess_questions(
                    self.data.test_questions)
                with open(self.config.test_qs_processed_path, 'wb') as f:
                    pickle.dump(self.data.test_qs_processed, f)


if __name__ == '__main__':
    train_based_QA = TrainBasedQA()
