from transformers import AutoImageProcessor, Dinov2Model
import torch
from PIL import Image
import os
from typing import List, Tuple, Dict, Any, Union, Optional
import torch.nn as nn
from .language import SentenceEmbedding
from .vision import ImageEmbedding
import torch.nn.functional as F
import torch
from transformers.activations import ACT2FN

class SiglipMLP(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class VLContrastHead(nn.Module):
    def __init__(self, vision_dimesion, text_dimension, device, target_dimension=512, linear=False):
        super(VLContrastHead, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear = linear
        if self.linear:
            self.vision_mapping_network = nn.Linear(vision_dimesion, target_dimension)
            self.text_mapping_network = nn.Linear(text_dimension, target_dimension)
        else:
            self.vision_mapping_network = SiglipMLP(vision_dimesion, target_dimension, target_dimension)
            self.text_mapping_network = SiglipMLP(text_dimension, target_dimension, target_dimension)

        self.vision_layer_norm = nn.LayerNorm(vision_dimesion)
        self.text_layer_norm = nn.LayerNorm(text_dimension)
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        self._initialize_weights()
    
    def _initialize_weights(self):
        if self.linear:
            torch.nn.init.xavier_uniform_(self.vision_mapping_network.weight)
            torch.nn.init.zeros_(self.vision_mapping_network.bias)
            torch.nn.init.xavier_uniform_(self.text_mapping_network.weight)
            torch.nn.init.zeros_(self.text_mapping_network.bias)
        else:
            # Xavier initialization for SiglipMLP layers
            for layer in [self.vision_mapping_network.fc1, self.vision_mapping_network.fc2,
                        self.text_mapping_network.fc1, self.text_mapping_network.fc2]:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        for layer in [self.vision_layer_norm, self.text_layer_norm]:
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        # Initialize logit_scale and logit_bias
        logit_scale_init = torch.log(torch.tensor(10.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(torch.tensor(-10.0))
        
    
    def forward(self, vision_embeddings=None, text_embeddings=None):

        if vision_embeddings is None and text_embeddings is None:
            raise ValueError("At least one of vision_embeddings and text_embeddings should be provided.")
        
        if vision_embeddings is not None:
            vision_embeddings = self.vision_layer_norm(vision_embeddings)
            vision_embeddings = self.vision_mapping_network(vision_embeddings) 
        else:
            vision_embeddings = None
        
        if text_embeddings is not None:
            text_embeddings = self.text_layer_norm(text_embeddings)
            text_embeddings = self.text_mapping_network(text_embeddings)
        else:
            text_embeddings = None

        if vision_embeddings is not None and text_embeddings is not None:
            norm_vision_embeddings = vision_embeddings / vision_embeddings.norm(p=2, dim=-1, keepdim=True)
            norm_text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
            logits_per_text = torch.matmul(norm_text_embeddings, norm_vision_embeddings.t()) * self.logit_scale.exp() + self.logit_bias
        else:
            logits_per_text = None
      
        return vision_embeddings, text_embeddings, logits_per_text
    

class VLContrastModel(nn.Module):
    def __init__(self, vision_model_name, text_model_name, device=None, vlhead_weights_path=None, linear=False):
        super(VLContrastModel, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_model = SentenceEmbedding(text_model_name, device)
        self.vision_model = ImageEmbedding(vision_model_name, device)
        self.vlhead = VLContrastHead(vision_dimesion=self.vision_model.model.config.hidden_size*2, text_dimension=self.text_model.model.config.hidden_size, device=self.device, linear=linear)

        if vlhead_weights_path:
            self.load_vlhead_weights(vlhead_weights_path)

    def freeze_except_vlhead(self):
        # Freeze vision model
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Freeze text model
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Do not freeze vlhead
        for param in self.vlhead.parameters():
            param.requires_grad = True

    def load_vlhead_weights(self, vlhead_weights_path):
        weights = torch.load(vlhead_weights_path, map_location=self.device)
        self.vlhead.load_state_dict(weights)
        print(f"Loaded VL head weights from {vlhead_weights_path}")
    
    def encode_image(self, image, normalize: bool = False):
        features = self.vision_model(image)
        vision_embeddings, _, _ = self.vlhead(vision_embeddings=features)
        return F.normalize(vision_embeddings, dim=-1) if normalize else vision_embeddings
    
    def encode_text(self, text, normalize: bool = False):
        features = self.text_model(text)
        _, text_embeddings, _ = self.vlhead(text_embeddings=features)
        return F.normalize(text_embeddings, dim=-1) if normalize else text_embeddings
    
    def forward(self, images=None, sentences=None):
        norm_vision_embeddings = self.encode_image(images, normalize=True)
        norm_text_embeddings = self.encode_text(sentences, normalize=True)
        # Log the sizes of embeddings
        logits_per_text = torch.matmul(norm_text_embeddings, norm_vision_embeddings.t()) * self.vlhead.logit_scale.exp() + self.vlhead.logit_bias
        return norm_vision_embeddings, norm_text_embeddings, logits_per_text
        
