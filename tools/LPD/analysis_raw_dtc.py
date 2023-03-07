import numpy as np
import mmcv
import pickle 
import os,sys
import cv2
import json
from PIL import Image
from collections import defaultdict,Mapping
from bidict import bidict
from random import seed,shuffle
from tqdm import tqdm
import shutil
from time import asctime
import shapely
from shapely.geometry import Polygon
from rich.progress import track
from easydict import EasyDict

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

type_counts_map = defaultdict(lambda: 0)


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

        if ifkeep_old and os.path.exists(self.ds_path):
            import shutil
            from time import asctime
            shutil.copy(self.ds_path,self.ds_path + f'.old.{asctime()}')

        with open(self.ds_path,'wb') as f:
            pickle.dump(self.ds,f)
        return



class DTCRAWAnalysis:
    def __init__(self,kv_id_file) -> None:
        with open(kv_id_file,'rb') as f:
            self.kv_id_ds = pickle.load(f)
            self.type_counts_map = defaultdict(lambda: 0)
    
    def type_statstics(self,):
        for id, fn in track(self.kv_id_ds['id_fn_mapping'].items()):
            img_fn = os.path.join(self.kv_id_ds['root_dir'], fn)
            label_fn = img_fn[:img_fn.rfind('.')] + '.txt'
            try:
                self.stat_label(label_fn)
            except Exception as e:
                    print(f'error in {label_fn}')
        return

    def stat_label(self, label_fn):
        if not os.path.exists(label_fn):
            return
        lines = open(label_fn,'r').readlines()
        for line in lines:
            ins_label = json.loads(line.strip())
            found_type_k =False

            # 为 legacy 中data定制
            # "Palate_visibleness":"Unrecognizable" 
            if (ins_label.get('Palate_visibleness','') == 'Unrecognizable'):
                lp_type = 'unrecognizable'
                self.type_counts_map[lp_type]+=1
                continue
            elif('vehicle_type' in ins_label.keys()):
                continue
            #
            for candi_key in [
                'vehiclePlate',
                'license_type',
                'license_type',
                'vehiclePlate'
            ]: 
                lp_type = ins_label.get(candi_key,'')
                if lp_type:
                    self.type_counts_map[lp_type]+=1
                    found_type_k=True
                    break
            if not found_type_k:
                raise(AssertionError(f'no key in keys: {ins_label.keys()}'))
        return


    def getdd(self):
        out_dict = {
            k:v for k,v in self.type_counts_map.items()
        }
        return out_dict



    
if __name__ == '__main__':
    kv_ds_fn = '/media/112new_sde/LPD/LPDAnnotations/ID_ImgFiles_KV'

    stastics_fn = '/media/112new_sde/LPD/LPDAnnotations/type_stats.pkl'
    ana = DTCRAWAnalysis(kv_ds_fn)
    ana.type_statstics()
    dd = ana.getdd()
    
    for k, v in dd.items():
        print(f"{k}:\t{v}")

    with open(stastics_fn,'wb') as f:
        pickle.dump(ana.getdd(),f)
