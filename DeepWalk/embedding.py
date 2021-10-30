import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import cpu_count
from random import shuffle, choice


class DeepWalk(object):
    def __init__(self, dimension=128, max_paths=10, path_len=10, window=5, hs=1):
        self.dimension = dimension  # Dimensions of word embeddings
        self.max_paths = max_paths  # Number of walks per node
        self.path_len = path_len    # Length of random walk
        self.window = window        # Window size for skipgram
        self.hs = hs                # 0 - Negative Sampling  1 - Hierarchical Softmax

    def embed(self, G):
        corpus = self.build_corpus(G)
        word_vec = self.generate_embeddings(corpus)

        return np.array([word_vec[word] for word in sorted(word_vec.wv.vocab, key=lambda x: self.sorted_node_key(x))])

    def sorted_node_key(self, x):
        try:
            return int(x)
        except:
            return x

    def random_walk(self, G, start_node):
        path = [str(start_node)]
        current = start_node
        
        while(len(path) < self.path_len):
            neighbors = list(G.neighbors(current))
            
            if(len(neighbors) == 0):
                break
            
            current = choice(neighbors)
            path.append(str(current))
            
        return path

    def build_corpus(self, G):
        corpus = list()
        nodes = list(G)

        for _ in range(self.max_paths):
            shuffle(nodes)
            corpus += [self.random_walk(G, i) for i in nodes]
    
        return corpus

    def generate_embeddings(self, corpus):
        model = Word2Vec(size=self.dimension, window=self.window, sg=1, min_count=0, hs=self.hs, workers=cpu_count())
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
        
        return model.wv
