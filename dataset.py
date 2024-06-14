import os
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import math
from scipy.spatial.distance import cosine
import time

class DatasetLoader():
    
    def __init__(self, path):
        self.path = path
        self.df = None

    def load_data(self,  type = 'csv'):
        self.df = pd.read_csv(self.path)
        return self.df
        
    # def clean_data(self):
        