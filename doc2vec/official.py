import multiprocessing
from pprint import pprint

from tqdm import tqdm

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

wiki = WikiCorpus("enwiki-latest-pages-articles.xml.bz2")


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument([c.decode("utf-8") for c in content], [title])


documents = TaggedWikiDocument(wiki)

pre = Doc2Vec(min_count=0)
pre.build_vocab(documents)
print('scan_vocab')

for num in range(0, 20):
    print('min_count: {}, size of vocab: '.format(num), pre.scale_vocab(min_count=num, dry_run=True)['memory']['vocab']/700)

cores = multiprocessing.cpu_count()

model = Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter =10, workers=cores),

model.build_vocab(documents)

model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

model.save('doc2vec.model')
model = models.Doc2Vec.load('doc2vec.model')

print(str(model))
pprint(model.docvecs.most_similar(positive=["Machine learning"], topn=20))
