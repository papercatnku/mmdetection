import sys
import os
import datetime
from torch.distributed import launch as distributed_launch
from volatile_configs_history import config


config = './configs/my_custom_configs/lpd_yolox_s.py'

distributed = 1
gpus = 4
dataset_style = 'coco' #'voc' #'coco'
master_port = 29500

os.environ["CUDA_VISIBLE_DEVICES"]=4,5,6,7

base_path = os.path.splitext(os.path.basename(config))[0]
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f'Training with: {config} @ {date}')


if distributed:
    sys.argv = [sys.argv[0], f'--nproc_per_node={gpus}', f'--master_port={master_port}',
                './tools/train.py', '--launcher=pytorch',
                config]

    distributed_launch.main()
else:
    from tools import train as train_mmdet
    sys.argv = [sys.argv[0], f'--gpus={gpus}', '--no-validate',
                f'{config}']

    args = train_mmdet.parse_args()
    train_mmdet.main(args)
#
