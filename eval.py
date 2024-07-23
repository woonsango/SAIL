from evaluation import (
    imagenet_eval,
    winoground_eval,
    coco_eval,
    update_results_json,
    extract_info_from_path,
    check_epoch_exists,
)
import argparse
from model import create_model
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation")
    parser.add_argument(
        "--task",
        type=str,
        choices=["imagenet", "COCO", "winoground"],
        default="imagenet",
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
        "--linear-align",
        default=False,
        action="store_true",
        help="Use linear projection head.",
    )
    parser.add_argument(
        "--linear-type",
        type=str,
        default="linear",
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
        "--images_dir",
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
    )

    return parser.parse_args()


def main():
    # load model, get device, decide eval dataset
    args = parse_args()
    # epoch_num = 1
    # training_info_str = "test"
    # model_prefix = "test"

    epoch_num, training_info_str, model_prefix = extract_info_from_path(args.head_weights_path)
    output_path = os.path.join(
        args.results_dir, args.task, model_prefix, f"{training_info_str}.json"
    )
    if check_epoch_exists(output_path, epoch_num):
        print(f"Epoch {epoch_num} already exists in {args.task}, skipping.")
        return
    model = create_model(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        head_weights_path=args.head_weights_path,
        linear_align=args.linear_align,
        linear_type = args.linear_type,
        target_dimension=args.target_dimension,
        device=args.device,
    )
    text_model_name = args.text_model.split("/")[-1]
    vision_model_name = args.vision_model.split("/")[-1]
    model.eval()

    # eval
    if args.task.lower() == "imagenet":
        results = imagenet_eval(
            model,
            bs=args.batch_size,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            images_dir=args.images_dir,
            save_dir=args.save_dir,
        )
    elif args.task.lower() == "coco":

        coco_root = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/val2017"
        coco_ann_file = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/annotations/captions_val2017.json"
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

    update_results_json(output_path, epoch_num, results)


if __name__ == "__main__":
    main()
