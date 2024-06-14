# Import required libraries
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from sentence_transformers import util
import torch
import numpy as np
from torch.nn.functional import normalize

class EmbeddingsModel():
    
    def __init__(self, model_name, api_key):
        self.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = self.api_key)
        self.model = AutoModel.from_pretrained(model_name, token = self.api_key)
    
    def mean_pooling(self):
        
        output = self.output[0]
        input_mask = self.attention_mask.unsqueeze(-1).expand(output.size()).float()
        return_val = torch.sum(output * input_mask,1) / torch.clamp(input_mask.sum(1), min = 1e-9)
        
        return return_val
    
    def get_embeddings(self, data):
        encoded_input = self.tokenizer(data, padding = True, truncation = True, return_tensors = 'pt')
        self.attention_mask = encoded_input['attention_mask']

        with torch.no_grad():
            self.output = self.model(**encoded_input)
        embeddings = self.mean_pooling()

        return normalize(embeddings)
    
    def get_similar_docs(self, data, topk = 3):
        results = []
        for doc in data:
            similarities = util.cos_sim(doc, data)
            similar_docs = torch.topk(similarities.flatten(), topk).indices
            results.append(similar_docs)
        return results
