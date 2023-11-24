import numpy as np
import mmcv
import pickle
import os,sys
import cv2
import json
from PIL import Image
from collections import defaultdict,Mapping,namedtuple
from random import seed,shuffle
from tqdm import tqdm
import shutil
from time import asctime
import shapely
from shapely.geometry import Polygon

from bidict import bidict
import dbm
from easydict import EasyDict

import rich
from rich.progress import track

empty_ds = EasyDict(
    next_id=0,
    root_dir='',
    id_fn_mapping=bidict()
)

def get_subfiles(path, length=None):
    out_ls = []
    def _func(x, ls=[]):
        for fn in os.listdir(x):
            full_path = os.path.join(x, fn)
            if os.path.isdir(full_path):
                _func(full_path, ls)
            else:
                ls.append(full_path)
            if length != None and len(out_ls) > length:
                return
        return
    _func(path, out_ls)
    return out_ls

img_post_ls=['jpg','png','jpeg',]

def makelabel_nm(x):
    return x[:x.rfind('.')] + '.txt'

class kvImgFilesMaintain:
    def __init__(self, ds_path,backend='pickle'):
        os.makedirs(os.path.dirname(ds_path), exist_ok=True)
        self.ds_path = ds_path
        if backend == 'pickle':
            if os.path.isfile(ds_path):
                self.ds = pickle.load(open(ds_path,'rb'),)
            else:
                self.ds = empty_ds.copy()
        else:
            print(f'Unsupportted backend: {backend}')

        return

    def add_dir(self,src_dir, img_prefix,ifkeep_old=True):


        self.ds['root_dir'] = img_prefix

        file_ls = get_subfiles(src_dir)

        img_fn_ls = filter(
            lambda x: x[x.rfind('.')+1:].lower() in img_post_ls, file_ls
        )

        invers_dict = self.ds['id_fn_mapping'].inverse

        img_fn_ls =list(img_fn_ls)
        for i,fn in track(enumerate(img_fn_ls)):
            relative_fn = fn[len(img_prefix):]

            if relative_fn in invers_dict.keys():
                continue
            self.ds['id_fn_mapping'][self.ds['next_id']] = relative_fn
            self.ds['next_id']+=1


        # self.ds['next_id'] += len(img_fn_ls)

        if ifkeep_old and os.path.exists(self.ds_path):
            import shutil
            from time import asctime
            shutil.copy(self.ds_path,self.ds_path + f'.old.{asctime()}')

        with open(ds_path,'wb') as f:
            pickle.dump(self.ds,f)

        return


if __name__ == '__main__':
    ds_path = '/media/112new_sde/LPD/LPDAnnotations/ID_ImgFiles_KV_sentry'
    cvt = kvImgFilesMaintain(ds_path)
    img_prefix = '/media/112new_sde/LPD/DTC_RAW/'

    for src_dir in [
        # '/media/112new_sde/LPD/DTC_RAW/Legacy',
        # '/media/112new_sde/LPD/DTC_RAW/DVR',
        # '/media/112new_sde/LPD/DTC_RAW/AVM',
        # '/media/112new_sde/LPD/DTC_RAW/BSD',
        # '/media/112new_sde/LPD/DTC_RAW/APA',
        # '/media/112new_sde/LPD/DTC_RAW/HK_Macau'
        '/media/112new_sde/LPD/DTC_RAW/AVM_SENTRY'
    ]:
        print(f'working on {src_dir}')
        cvt.add_dir(src_dir,img_prefix)
        print(f'{src_dir} done.')

    # cvt.add_dir(src_dir,img_prefix)






