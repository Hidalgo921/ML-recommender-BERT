import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GridSearchCV

class TSNE_Wrapper(BaseEstimator, TransformerMixin):

    def __init__(self,perplexity, n_iter, random_state, n_components =2):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data):
        self.data = data
        self.tsne_model = TSNE(self.n_components, self.perplexity, self.n_iter, self.random_state)
        embeddings_2d = self.tsne_model.fit_transform(data)
        return embeddings_2d
    
    def score(self, data):
        return 0.0
    
    def tune_model(self, param_grid):
        grid_search = GridSearchCV(self.tsne_model, param_grid, cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(self.data)
        param_dict = grid_search.best_params_
        n_iter, perplexity = param_dict.values()
        self.n_iter = n_iter
        self.perplexity = perplexity
    
class Clust_Wrapper():

    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def get_clusters(self, data):
        clust_model = KMeans(self.n_clusters)
        labels = clust_model.fit_predict(data)
        return labels