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
        choices=["imagenet", "coco", "winoground"],
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
        print(f"Epoch {epoch_num} already exists in {output_path}, skipping.")
        return
    model = create_model(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        head_weights_path=args.head_weights_path,
        linear_align=args.linear_align,
        device=args.device,
    )

    model.eval()

    # eval
    if args.task == "imagenet":
        results = imagenet_eval(
            model,
            bs=args.batch_size,
            text_model_name=args.text_model,
            vision_model_name=args.vision_model,
            images_dir=args.images_dir,
        )
    elif args.task == "coco":
        # coco_root = "/home/mila/q/qian.yang/scratch/coco2017/val2017"
        # coco_ann_file = "/home/mila/q/qian.yang/scratch/coco2017/2017/annotations/captions_val2017.json"

        coco_root = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/val2017"
        coco_ann_file = "/home/mila/l/le.zhang/scratch/datasets/coco/2017/annotations/captions_val2017.json"
        results = coco_eval(
            model,
            bs=args.batch_size,
            coco_root=coco_root,
            coco_ann_file=coco_ann_file,
            text_model_name=args.text_model,
            vision_model_name=args.vision_model,
        )
    elif args.task == "winoground":
        results = winoground_eval(
            model,
            text_model_name=args.text_model,
            vision_model_name=args.vision_model,
        )

    update_results_json(output_path, epoch_num, results)


if __name__ == "__main__":
    main()
