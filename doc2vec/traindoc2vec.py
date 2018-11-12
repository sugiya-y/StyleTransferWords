import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts, get_tmpfile


def main():
    with open('parsed.pickle', 'rb') as f:
        enwiki = pickle.load(f)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(enwiki)]
    model = Doc2Vec(documents, vector_size=200, window=5, min_count=5, workers=16)

    # fname = get_tmpfile("my_doc2vec_model")
    model.save("./enwiki.model")
    model = Doc2Vec.load("./enwiki.model")


if __name__ == '__main__':
    main()
