DATADIR = {
    'LLaVA558K' : {
        'annotation':'/home/mila/l/le.zhang/scratch/light_align/data/blip_laion_cc_sbu_558k.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/light_align/data/image'
        },
    'ALLaVALAION' : {
        # note that some images are missing in the original repo, thus we filter out invalid items
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/allava_laion/ALLaVA-Caption-LAION-4V-valid.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets'
        },
    'ALLaVAVFLAN': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/allava_vflan/ALLaVA-Caption-VFLAN-4V.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets'
        },
    'coco': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment/jsons/coco.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment'
        },
    'sam': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment/jsons/sam.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment'
        },
    'Sharegpt4vllava': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment/jsons/llava_pretrain.json',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/Cambrian-Alignment'
        },
    'dreamclipcc3m': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M/cc3m_3long_1raw_captions_filterd.csv',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M'
        },
    'dreamclipcc12m': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M/cc12m_3long_3short_1raw_captions_url_path_filtered.csv',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M'
        },
    'dreamclipcc12mhf': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M/cc12m_3long_3short_1raw_captions_url_path_hf_filtered.csv',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/DownloadCC3M'
        },
    'laion30m': {
        'annotation':'/home/mila/l/le.zhang/scratch/datasets/LAION/30M_laion_synthetic_filtered_large_with_path_filtered.csv',
        'imagedir':'/home/mila/l/le.zhang/scratch/datasets/LAION'
        },
    
}