from evaluation.seg_eval import segmentation_eval

text_model_name = "Alibaba-NLP/gte-large-en-v1.5"
vision_model_name = "facebook/dinov2-large"
head_weights_path = "logs/dreamclip30m_gtendinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2/checkpoints/epoch_30.pt"
linear_type = "star"
target_dimension = 1024
device = "cuda"
use_gmp = False
gmp_groups = 1
task_config = "evaluation/ClearCLIP/configs/cfg_voc20.py"
task_config = "evaluation/ClearCLIP/configs/cfg_ade20k.py"
save_dir = "evaluation/backbone_features"

segmentation_eval(
    text_model_name,
    vision_model_name,
    head_weights_path,
    linear_type,
    target_dimension,
    device,
    use_gmp,
    gmp_groups,
    task_config,
    save_dir,
    visualize=False,
)
