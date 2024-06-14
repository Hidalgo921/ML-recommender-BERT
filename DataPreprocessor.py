import numpy as np
import pandas as pd
import re
import nltk

class DataPreprocessor():

    def __init__(self,df):
        self.df = df.copy()

    def remove_stop(self, stop_words, col):
        lst = self.df[col].tolist()

        processed_lst = []
        for record in lst:
            tokens = [i.split(' ') for i in str(record).split(".")] #split sentence into terms
            tokens = [[re.sub(r'[^a-zA-Z0-9]', '', word) for word in token] for token in tokens] #replace any non-alphanumeric characters w/ blank space
            tokens = [[word.lower() for word in token if word.lower() not in stop_words and word != ''] for token in tokens] #remove stop words & empty strings
            tokens = [token for token in tokens if len(token)>0] #remove any sentence records that are empty
            processed_rec = [' '.join(token) for token in tokens] #join string terms back into cleaned sentences
            processed_rec = " ".join(sent for sentence in processed_rec for sent in sentence.split()) #clean sentences into single desc
            processed_lst.append(processed_rec)
        return processed_lst