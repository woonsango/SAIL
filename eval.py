from evaluation import (
    imagenet_eval,
    winoground_eval,
    coco_eval,
    SugarCrepe_eval,
    update_results_json,
    extract_info_from_path,
    check_epoch_exists,
    segmentation_eval,
)
import argparse
from model import create_model
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation")
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "imagenetv2",
            "COCO",
            "winoground",
            "imagenetv1",
            "sugar_crepe",
            "segmentation",
        ],
        default="imagenetv1",
        help="Task",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=None,
        help="e.g sentence-transformers/all-mpnet-base-v2. If provided, will load a text model and use it for text embeddings.",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="e.g facebook/dinov2-base. If provided, will load a vision model and use it for image embeddings.",
    )
    parser.add_argument(
        "--head-weights-path",
        type=str,
        default=None,
        # required=True,
        help="Path to head weight",
    )
    parser.add_argument(
        "--linear-type",
        type=str,
        default="star",
        help="Type of linear layer to use.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/mila/l/le.zhang/scratch/light_align/evaluation/eval_result",
        help="Path to results file",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="/home/mila/l/le.zhang/scratch/datasets",
        help="Path to images",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/mila/l/le.zhang/scratch/light_align/evaluation/backbone_features",
        help="Path to images",
    )
    parser.add_argument(
        "--target-dimension",
        type=int,
        default=512,
        help="Dimension of text embeddings. Default set to 768 for all-mpnet-base-v2.",
    ),
    parser.add_argument(
        "--use_gmp",
        default=False,
        action="store_true",
        help="Use grouped mean pooling for image features.",
    )
    parser.add_argument(
        "--gmp_groups",
        type=int,
        default=512,
        help="Number of groups for grouped mean pooling.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite existing results.",
    )
    parser.add_argument(
        "--seg_task_config",
        type=str,
        default="/home/mila/q/qian.yang/Light_Align/evaluation/ClearCLIP/configs/cfg_ade20k.py",
        help="Task for segmentation evaluation",
    )

    parser.add_argument(
        "--visualize_segmentation",
        default=False,
        action="store_true",
        help="Visualize segmentation results.",
    )
    args = parser.parse_args()

    # Overide args with model_config.yaml
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(args.head_weights_path)), "model_config.yaml"
    )
    if os.path.exists(config_file):
        print(f"Loading model config from {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args


def get_output_path_and_check_epoch(args, epoch_num, training_info_str, model_prefix):
    output_path = os.path.join(
        args.results_dir, args.task, model_prefix, f"{training_info_str}.json"
    )

    if check_epoch_exists(output_path, epoch_num) and not args.overwrite:
        print(f"Epoch {epoch_num} already exists in {args.task}, skipping.")
        return None
    elif check_epoch_exists(output_path, epoch_num) and args.overwrite:
        print(f"Epoch {epoch_num} already exists in {args.task}, overwriting.")

    return output_path


def main():
    args = parse_args()
    # for debug
    # epoch_num = 1
    # training_info_str = "test"
    # model_prefix = "test"

    epoch_num, training_info_str, model_prefix = extract_info_from_path(
        args.head_weights_path
    )
    try:
        output_path = get_output_path_and_check_epoch(
            args, epoch_num, training_info_str, model_prefix
        )
    except:
        output_path = os.path.join(
            args.results_dir,
            args.task,
            model_prefix,
            f"{training_info_str}{'gmp_groups'+ str(args.gmp_groups) if args.use_gmp else ''}.json",
        )
    if check_epoch_exists(output_path, epoch_num) and not args.overwrite:
        print(f"Epoch {epoch_num} already exists in {args.task}, skipping.")
        return

    model = create_model(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        head_weights_path=args.head_weights_path,
        linear_type=args.linear_type,
        target_dimension=args.target_dimension,
        device=args.device,
        use_gmp=args.use_gmp,
        gmp_groups=args.gmp_groups,
        test=True,
    )
    text_model_name = args.text_model.split("/")[-1]
    vision_model_name = args.vision_model.split("/")[-1]
    model.eval()

    # eval
    if args.task.lower() == "imagenetv1":
        # Check if the ImageNet folders exist, if not create them and download the data
        imagenet_dir = os.path.join(args.dataset_root_dir, "imagenet")
        if not os.path.exists(imagenet_dir):
            os.makedirs(imagenet_dir)
            print(f"Created directory: {imagenet_dir}")

            # Download ImageNet validation images
            val_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
            val_file = os.path.join(imagenet_dir, "ILSVRC2012_img_val.tar")
            if not os.path.exists(val_file):
                print(f"Downloading ImageNet validation images to {val_file}")
                os.system(f"wget {val_url} -O {val_file}")
            else:
                print(f"ImageNet validation images already exist at {val_file}")

            # Download ImageNet devkit
            devkit_url = (
                "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
            )
            devkit_file = os.path.join(imagenet_dir, "ILSVRC2012_devkit_t12.tar.gz")
            if not os.path.exists(devkit_file):
                print(f"Downloading ImageNet devkit to {devkit_file}")
                os.system(f"wget {devkit_url} -O {devkit_file}")
            else:
                print(f"ImageNet devkit already exists at {devkit_file}")
        results = imagenet_eval(
            model,
            bs=args.batch_size,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            images_dir=imagenet_dir,
            save_dir=args.save_dir,
            version="v1",
        )
    elif args.task.lower() == "imagenetv2":
        results = imagenet_eval(
            model,
            bs=args.batch_size,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            images_dir=args.dataset_root_dir,
            save_dir=args.save_dir,
            version="v2",
        )
    elif args.task.lower() == "coco":

        coco_root = os.path.join(args.dataset_root_dir, "coco", "2017", "val2017")
        coco_ann_file = os.path.join(
            args.dataset_root_dir,
            "coco",
            "2017",
            "annotations",
            "captions_val2017.json",
        )
        results = coco_eval(
            model,
            bs=args.batch_size,
            coco_root=coco_root,
            coco_ann_file=coco_ann_file,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            k_vals=[1, 5, 10],
            save_dir=args.save_dir,
        )
    elif args.task.lower() == "winoground":
        results = winoground_eval(
            model,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            save_dir=args.save_dir,
        )
    elif args.task.lower() == "sugar_crepe":
        coco_root = os.path.join(args.dataset_root_dir, "coco", "2017", "val2017")
        results = SugarCrepe_eval(
            model,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            images_dir=coco_root,
            save_dir=args.save_dir,
            bs=args.batch_size,
        )
    elif args.task.lower() == "segmentation":
        results = segmentation_eval(
            text_model_name=args.text_model,
            vision_model_name=args.vision_model,
            head_weights_path=args.head_weights_path,
            linear_type=args.linear_type,
            target_dimension=args.target_dimension,
            device=args.device,
            use_gmp=args.use_gmp,
            gmp_groups=args.gmp_groups,
            task_config=args.seg_task_config,
            save_dir=args.save_dir,
            visualize=args.visualize_segmentation,
            # precision='fp16',
        )
    update_results_json(output_path, epoch_num, results)


if __name__ == "__main__":
    main()
