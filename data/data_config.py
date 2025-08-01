import os
SCRATCH = "/home"

DATADIR = {

    'dreamclipcc3m': {
        'annotation':f'{SCRATCH}/datasets/DownloadCC3M/cc3m_3long_1raw_captions_filtered.csv',
        'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
        },
    'dreamclipcc12m': {
        'annotation':f'{SCRATCH}/datasets/DownloadCC3M/cc12m_3long_3short_1raw_captions_url_filtered.csv',
        'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
        },
    'yfcc15m': {
        'annotation':f'{SCRATCH}/datasets/DownloadCC3M/yfcc15m_3long_3short_1raw_captions_url_filtered.csv',
        'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
        },
    'coco2017': {
        'annotation':f'{SCRATCH}/dataset/coco2017/annotations/captions_train2017.json',
        'imagedir':f'{SCRATCH}/dataset/coco2017/train2017'
        },
    
    
}