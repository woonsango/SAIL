import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional, Union

SentenceEmbeddingStrategy={
    'mean': ['sentence-transformers/all-mpnet-base-v2'],
    'BOS' : ['Alibaba-NLP/gte-large-en-v1.5'],
    "LastToken": ['Alibaba-NLP/gte-Qwen2-7B-instruct',"Alibaba-NLP/gte-Qwen2-1.5B-instruct"],
}


def get_embedding_strategy(model_name):
    for strategy, models in SentenceEmbeddingStrategy.items():
        if model_name in models:
            return strategy
    return None

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class SentenceEmbedding(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', device=None):
        super(SentenceEmbedding, self).__init__()
        self.model_name = model_name
        self.ses = get_embedding_strategy(model_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
    
    def mean_pooling(self, model_output: torch.Tensor, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_sentence_embeddings(self, sentences: List[str]):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(self.device)
        return self.forward(encoded_input)
    
    def forward(self, inputs):
        # Compute token embeddings
        model_output = self.model(**inputs)
        if self.ses == 'mean':
            sentence_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
        elif self.ses == 'BOS':
            sentence_embeddings = model_output.last_hidden_state[:,0]
        elif self.ses == 'LastToken':
            sentence_embeddings = last_token_pool(model_output.last_hidden_state, inputs['attention_mask'])
        else:
            raise NotImplementedError(f"Strategy {self.ses} not implemented")
      
        return sentence_embeddings

# Usage example
if __name__ == "__main__":
    sentences = ['This is an example sentence', 'Each sentence is converted']
    
    embedder = SentenceEmbedding()
    sentence_embeddings = embedder.get_sentence_embeddings(sentences)
    
    print("Sentence embeddings:")
    print(sentence_embeddings)
    print(sentence_embeddings.shape)
