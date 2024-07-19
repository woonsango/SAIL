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
from .linear import StarMLP, SiglipMLP, SwiGLU



class VLContrastHead(nn.Module):
    def __init__(
        self,
        vision_dimesion: int,
        text_dimension: int,
        target_dimension: int,
        linear_align: bool = False,
        linear_type: str = "linear",
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super(VLContrastHead, self).__init__()
        self.cast_dtype = cast_dtype
        self.linear_align = linear_align
        self.linear_type = linear_type
        if linear_type == "star":
            LinearLayer = StarMLP
        elif linear_type == "mlp":
            LinearLayer = SiglipMLP
        elif linear_type == "swiglu":
            LinearLayer = SwiGLU
        else:
            LinearLayer = nn.Linear
        activation_fn = nn.ReLU6()
        self.vision_mapping_network = LinearLayer(vision_dimesion, target_dimension, activation=activation_fn)
        self.text_mapping_network = LinearLayer(text_dimension, target_dimension, activation=activation_fn)
        
        if not self.linear_align:
            self.shared_layer_norm = nn.LayerNorm(target_dimension)
            self.sigmoid = LinearLayer(target_dimension, 1)

        self.vision_layer_norm = nn.LayerNorm(vision_dimesion)
        self.text_layer_norm = nn.LayerNorm(text_dimension)
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Initialize logit_scale and logit_bias
        logit_scale_init = torch.log(torch.tensor(10.0))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(torch.tensor(-10.0))

    def forward(self, image_features=None, text_features=None, compute_logits=False):

        if image_features is None and text_features is None:
            raise ValueError(
                "At least one of image_features and text_features should be provided."
            )
        
        if image_features is not None:
            image_features = image_features.to(self.cast_dtype)
            image_features = self.vision_layer_norm(image_features)
            image_features = self.vision_mapping_network(image_features)
           
        if text_features is not None:
            text_features = text_features.to(self.cast_dtype)
            text_features = self.text_layer_norm(text_features)
            text_features = self.text_mapping_network(text_features)
            

        if not self.linear_align:
            image_features = self.shared_layer_norm(image_features) # n,d   
            text_features = self.shared_layer_norm(text_features) # n,d 
            B, _ = image_features.shape
            logits_per_text = self.sigmoid(image_features.unsqueeze(1).expand(-1, B, -1) * text_features.unsqueeze(1).expand(-1, B, -1)).squeeze(-1) # n,n

            
            
        if image_features is not None and text_features is not None and compute_logits:
            logits_per_text = (
                torch.matmul(F.normalize(text_features, dim=-1), F.normalize(image_features, dim=-1).t()) * self.logit_scale.exp()
                + self.logit_bias
            )
        else:
            logits_per_text = None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.logit_scale.exp(),
            "logit_bias": self.logit_bias,
        }


class VLContrastModel(nn.Module):
    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        target_dimension: int ,
        vlhead_weights_path: str = None,
        linear_align: bool = False,
        linear_type: str = "linear",
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super(VLContrastModel, self).__init__()
        self.text_model = SentenceEmbedding(text_model_name)
        self.vision_model = ImageEmbedding(vision_model_name)
        self.vlhead = VLContrastHead(
            vision_dimesion = self.vision_model.model.config.hidden_size * 2,
            text_dimension = self.text_model.model.config.hidden_size,
            target_dimension = target_dimension,
            linear_align = linear_align,
            linear_type = linear_type,
            cast_dtype = cast_dtype,
        )

        if vlhead_weights_path is not None:
            self.load_vlhead_weights(vlhead_weights_path)


    def freeze_except_vlhead(self):
        self._freeze_parameters(self.vision_model)
        self._freeze_parameters(self.text_model)
        self._unfreeze_parameters(self.vlhead)

    def _freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def load_vlhead_weights(self, vlhead_weights_path):
        weights = torch.load(vlhead_weights_path)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        self.vlhead.load_state_dict(weights, strict=False)
        print(f"Loaded VL head weights from {vlhead_weights_path}")

    def encode_image(self, image, normalize: bool = False):
        features = self.vision_model(image)
        outputs = self.vlhead(image_features=features)
        image_features = outputs["image_features"]
        return F.normalize(image_features, dim=-1) if normalize else image_features

    def encode_image_head(self, pre_encode_image_featuers, normalize: bool = False):
        outputs = self.vlhead(image_features=pre_encode_image_featuers)
        image_features = outputs["image_features"]
        return F.normalize(image_features, dim=-1) if normalize else image_features

    def encode_image_return_encodedFeatures(self, image, normalize: bool = False):
        features = self.vision_model(image)
        outputs = self.vlhead(image_features=features)
        image_features = outputs["image_features"]
        return (
            F.normalize(image_features, dim=-1) if normalize else image_features
        ), features

    def encode_text(self, text, normalize: bool = False):
        features = self.text_model(text)
        outputs = self.vlhead(text_features=features)
        text_features = outputs["text_features"]
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def encode_text_head(self, pre_encode_model_featuers, normalize: bool = False):
        outputs = self.vlhead(text_features=pre_encode_model_featuers)
        text_features = outputs["text_features"]
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def encode_text_return_encodedFeatures(self, text, normalize: bool = False):
        features = self.text_model(text)
        outputs = self.vlhead(text_features=features)
        text_features = outputs["text_features"]
        return (
            F.normalize(text_features, dim=-1) if normalize else text_features
        ), features

    def forward(self, images=None, texts=None):
        norm_image_features = self.encode_image(images, normalize=True)
        norm_text_features = self.encode_text(texts, normalize=True)
        # Log the sizes of embeddings
        logits_per_text = (
            torch.matmul(norm_text_features, norm_image_features.t())
            * self.vlhead.logit_scale.exp()
            + self.vlhead.logit_bias
        )
        return {
            "image_features": norm_image_features,
            "text_features": norm_text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.vlhead.logit_scale.exp(),
            "logit_bias": self.vlhead.logit_bias,
        }

    def forward_return_encoded(self, images=None, texts=None):
        norm_image_features, encoded_image_features = (
            self.encode_image_return_encodedFeatures(images, normalize=True)
        )
        norm_text_features, encoded_text_features = (
            self.encode_text_return_encodedFeatures(texts, normalize=True)
        )
        # Log the sizes of embeddings
        logits_per_text = (
            torch.matmul(norm_text_features, norm_image_features.t())
            * self.vlhead.logit_scale.exp()
            + self.vlhead.logit_bias
        )
        return {
            "image_features": norm_image_features,
            "text_features": norm_text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.vlhead.logit_scale.exp(),
            "logit_bias": self.vlhead.logit_bias,
            "encoded_image_features": encoded_image_features,
            "encoded_text_features": encoded_text_features,
        }

    def forward_head(self, images=None, texts=None):
        norm_image_features = self.encode_image_head(images, normalize=True)
        norm_text_features = self.encode_text_head(texts, normalize=True)
        # Log the sizes of embeddings
        logits_per_text = (
            torch.matmul(norm_text_features, norm_image_features.t())
            * self.vlhead.logit_scale.exp()
            + self.vlhead.logit_bias
        )
        return {
            "image_features": norm_image_features,
            "text_features": norm_text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.vlhead.logit_scale.exp(),
            "logit_bias": self.vlhead.logit_bias,
        }
