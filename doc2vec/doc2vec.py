import pdb

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts, get_tmpfile

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

fname = get_tmpfile("my_doc2vec_model")
model.save("./mydoc2vec.model")
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

model.delete_temporary_training_data(
    keep_doctags_vectors=True, keep_inference=True)

vector = model.infer_vector(["system", "response"])

pdb.set_trace()
