import pandas as pd
import numpy as np
import EmbeddingModel
from scipy.spatial.distance import cosine


class BaselineModel():

    def __init__(self, file_path, model):
        self.df = pd.read_csv(file_path)
        self.model = model

    def get_sample(self, sample_size = 100):
        max_val = len(self.df['source_doc'].unique())
        self.sample_indices = np.random.randint(0,high = max_val, size = sample_size)
        self.sample_df = self.df[self.df['source_doc'].isin(self.sample_indices)]

    def embed_data(self):
        col1_embeddings = self.model.get_embeddings(self.sample_df['source_Description'].tolist())
        col2_embeddings = self.model.get_embeddings(self.sample_df['matched_Description'].tolist())
        
        similarities = [1- cosine(vec, col2_embeddings[i]) for i, vec in enumerate(col1_embeddings)]
        return similarities
            
