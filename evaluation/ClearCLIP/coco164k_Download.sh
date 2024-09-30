#!/bin/bash
# download
# mkdir coco_stuff164k && cd coco_stuff164k
# wget http://images.cocodataset.org/zips/train2017.zip -P /home/mila/q/qian.yang/scratch/coco_stuff164k
# wget http://images.cocodataset.org/zips/val2017.zip -P /home/mila/q/qian.yang/scratch/coco_stuff164k
# wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip -P /home/mila/q/qian.yang/scratch/coco_stuff164k

# # unzip
# unzip /home/mila/q/qian.yang/scratch/coco_stuff164k/train2017.zip -d /home/mila/q/qian.yang/scratch/coco_stuff164k/images
# unzip /home/mila/q/qian.yang/scratch/coco_stuff164k/val2017.zip -d /home/mila/q/qian.yang/scratch/coco_stuff164k/images
# unzip /home/mila/q/qian.yang/scratch/coco_stuff164k/stuffthingmaps_trainval2017.zip -d /home/mila/q/qian.yang/scratch/coco_stuff164k/annotations

# --nproc means 8 process for conversion, which could be omitted as well.
python /home/mila/q/qian.yang/Light_Align/mmdetection/tools/dataset_converters/coco_stuff164k.py /home/mila/q/qian.yang/scratch/coco_stuff164k --nproc 8