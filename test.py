from nltk.stem.wordnet import WordNetLemmatizer


from enum import Enum, unique
@unique
class DocType(Enum):
    train = 0,
    dev = 1,
    test = 2

if DocType.train == DocType.train:
    print 1111

print DocType.train