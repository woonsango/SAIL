1. Please follow the Instructions in https://github.com/mc-lan/ClearCLIP to install dependencies, mainly the mmseg and mmengine.

2.  Now we support PASCAL VOC, ADE20k, VOC20. For other projects, I am preparing the datasets. ClearCLIP uses:

- With background class: PASCAL VOC, PASCAL Context, Cityscapes, ADE20k, and COCO-Stuff164k,

- Without background class: VOC20, Context59 (i.e., PASCAL VOC and PASCAL Context without the background category), and COCO-Object. 

3. I grant you the permission to /home/mila/q/qian.yang/scratch/segmentation_datasets, you should be able to access this dir.

- If you cannot access, please follow https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md to download PASCAL VOC and ADE20k

4. When Evaluation, set --task segmentation and --seg_task_config path_to_the_config (under ./evaluation/ClearCLIP/configs)

- You need to change the image_root in the configs if you canno access to /home/mila/q/qian.yang/scratch/segmentation_datasets