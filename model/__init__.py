from .sail_model import AlignmentLayer, SAILModel, ShareLockAlignmentLayer, AlignmentLayer_custom, ShareLockAlignmentLayer_custom
from .loss import ClipLoss, SigLipLoss, BarlowTwinsLoss
from .vision_model import ImageEmbedding
from .language_model import SentenceEmbedding
from typing import Union, Optional
import torch
import os


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    elif precision == "fp32":
        cast_dtype = torch.float32
    return cast_dtype


def create_model(
        text_model_name:Optional[str] = None, 
        vision_model_name:Optional[str] = None, 
        head_weights_path:Optional[str] = None,
        vision_dimesion:int = 1536,
        text_dimension:int = 768,
        target_dimension:int = 512,
        precision: str = 'fp32', 
        device: Union[str, torch.device] = 'cpu', 
        linear_type: str = 'star',
        logit_scale: float = 20.0,
        logit_bias: float = -10.0,
        agg_mode: str = 'concat',
        width_factor: int = 8,
        sharelock: bool = False,
        sail_model: bool = False,
        only_text: bool = False
):  
    if isinstance(device, str):
        device = torch.device(device)

    LayerClass = ShareLockAlignmentLayer if sharelock else AlignmentLayer

    cast_dtype = get_cast_dtype(precision)
    if sail_model:
        print("use SAILModel")
        if sharelock:
            print("use sharelock")
        model = SAILModel(
            text_model_name=text_model_name, 
            vision_model_name=vision_model_name, 
            target_dimension=target_dimension, 
            vlhead_weights_path=head_weights_path, 
            linear_type=linear_type, 
            cast_dtype=cast_dtype, 
            agg_mode=agg_mode,
            width_factor=width_factor,
            sharelock=sharelock,
            only_text=only_text,
        )
    else:
       print("don't use SAILModel")
       if sharelock:
            print("use sharelock")
       model = LayerClass(
            vision_dimesion, 
            text_dimension, 
            target_dimension, 
            linear_type=linear_type, 
            cast_dtype=cast_dtype, 
            logit_scale=logit_scale, 
            logit_bias=logit_bias, 
            width_factor=width_factor,
            only_text=only_text
        )
    model.to(device=device)
    return model

def create_model_custom(
        vision_dimesion:int = 1536,
        text_dimension:int = 768,
        target_dimension:int = 512,
        precision: str = 'fp32', 
        device: Union[str, torch.device] = 'cpu', 
        logit_scale: float = 20.0,
        logit_bias: float = -10.0,
        custom_Layer = None,
):
    if isinstance(device, str):
        device = torch.device(device)

    cast_dtype = get_cast_dtype(precision)

    model = AlignmentLayer_custom(
            vision_dimesion, 
            text_dimension, 
            target_dimension, 
            cast_dtype=cast_dtype, 
            logit_scale=logit_scale, 
            logit_bias=logit_bias, 
            custom_Layer=custom_Layer
        )
    model.to(device=device)
    return model

def create_model_CLIP(
        vision_dimesion:int = 1536,
        text_dimension:int = 768,
        target_dimension:int = 512,
        precision: str = 'fp32', 
        device: Union[str, torch.device] = 'cpu', 
        logit_scale: float = 20.0,
        logit_bias: float = -10.0,
        custom_Layer = None,
        alignmentLayer = None,
):
    if isinstance(device, str):
        device = torch.device(device)

    model = alignmentLayer
    model.to(device=device)
    return model

def create_model_LiT(
        vision_dimesion:int = 1536,
        text_dimension:int = 768,
        target_dimension:int = 512,
        precision: str = 'fp32', 
        device: Union[str, torch.device] = 'cpu', 
        logit_scale: float = 20.0,
        logit_bias: float = -10.0,
        custom_Layer = None,
        alignmentLayer = None,
):
    if isinstance(device, str):
        device = torch.device(device)

    model = alignmentLayer
    model.to(device=device)
    return model

def create_model_ShareLock(
        vision_dimesion:int = 1536,
        text_dimension:int = 768,
        target_dimension:int = 512,
        precision: str = 'fp32', 
        device: Union[str, torch.device] = 'cpu', 
        logit_scale: float = 20.0,
        logit_bias: float = -10.0,
        width_factor: int = 8,
        sharelock: bool = False,
        custom_Layer = None,
):  
    if isinstance(device, str):
        device = torch.device(device)

    cast_dtype = get_cast_dtype(precision)


    model = ShareLockAlignmentLayer_custom(
        vision_dimesion, 
        text_dimension, 
        target_dimension, 
        cast_dtype=cast_dtype, 
        logit_scale=logit_scale, 
        logit_bias=logit_bias, 
        width_factor=width_factor,
        custom_Layer=custom_Layer
    )
    model.to(device=device)
    return model


def create_loss(args):
    if args.siglip:
        print("Using SigLip loss")
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        print("Using Clip (infoNCE) loss")
        return ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
