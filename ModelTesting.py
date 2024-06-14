import numpy as np
from EmbeddingModel import EmbeddingsModel
import time


#testing
class TestSuite():

    def __init__(self, test_cases= None):
        self.test_cases = test_cases if test_cases is not None else {}

    def find_stop_words(self, stopwords, rec):
        tokens = [i.split(' ') for i in str(rec).split(".")]
        sw_lst = [word for word in stopwords if word in [i.lower() for token in tokens for i in token]] 
        return sw_lst

    def test_stop_words(self, stopwords,lst):
        stop_words = [i for i in set(stopwords.words('english'))]
        processed_lst = []
        for record in lst:
            sw_lst = self.find_stop_words(stop_words, record)
            processed_lst.append(sw_lst)
        assert all([len(i) == 0 for i in processed_lst])
             
    def test_removed_records(self, df2_length, filt_length):
        self.filtered_length = df2_length
        assert df2_length == filt_length
    
    def test_embedding_consistency(self, model, data):
        self.data = data
        self.model = model
        embedding1 = self.model.get_embeddings(self.data)
        embedding2 = self.model.get_embeddings(self.data)
        assert np.allclose(embedding1, embedding2)
    
    # def test_embedding_robustness(self):
    #     embedding1 = self.model.get_embeddings(self.data)
    #     embedding2 = self.model.get_embeddings(self.data)
    #     assert np.allclose(embedding1, embedding2)

    def test_embedding_processing(self, batch_size, processing_goal):
        batches = len(range(0,self.filtered_length, batch_size))
        start_time = time.time()
        encoded_rec = self.model.get_embeddings(self.data)
        process_time = time.time() - start_time
        total_time = process_time *batches
        assert total_time <= processing_goal

    def test_embedding_lengths(self, embeddings):
        self.embedding_length = len(embeddings[0])
        assert np.array(embeddings).shape == (self.filtered_length, self.embedding_length)

    def test_neighbor_lengths(self, input_dict, k_neighbors):
        assert all([len(v) == k_neighbors for v in input_dict.values()])
        assert self.filtered_length == len(input_dict)

    def run_tests(self):
        return_results = []
        for test, params in self.test_cases.items():
            try:
                test(**params)
                result = f'{test.__name__} Passed'
                return_results.append(result)
            except Exception as e:
                result = f'{test.__name__} Failed'
                return_results.append(result)
        return return_results
