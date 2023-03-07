import numpy as np
import mmcv
import pickle 
import os,sys
import cv2
import json
from PIL import Image
from collections import defaultdict
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

img_post_ls=['jpg','png','jpeg',]

def makelabel_nm(x):
    return x[:x.rfind('.')] + '.txt' 


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

def safeimread(src_fn, flags=cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION):
    return cv2.imdecode(np.fromfile(src_fn, dtype=np.uint8), flags=flags)

def pt_tolist(x): return [float(x) for x in x.split(',')]

def bbox_to_quad(bbox):
            xmin, ymin, xmax, ymax = bbox
            quad = [
                xmin, ymin,  # left top
                xmax, ymin,  # right top
                xmax, ymax,  # right bottom
                xmin, ymax,  # right top
                ]
            return quad


def quad_to_bbox(quad):
    bbox = [
            min(quad[::2]),
            min(quad[1::2]),
            max(quad[::2]),
            max(quad[1::2])]
    return bbox

def bbox_to_xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3]-bbox[1]]
    
def quad_to_xywh(quad):
    return bbox_to_xywh(quad_to_bbox(quad))

class DTCRAWCvtTool:
    def __init__(
            self,
            img_prefix='',
            existed_annotattion_file='',
            categories=['LP'],
            key_id_mapping=None,
            check_path = './tocheck.txt',
            ) -> None:
        # annotation dict keys: images, annotations, categories
        # aux keys: img_prefix, next_image_id, next_anno_id
        self.existed_annotattion_file = existed_annotattion_file
        self.categories = categories
        self.key_id_mapping = key_id_mapping
        if not self.key_id_mapping:
            self.key_id_mapping = defaultdict(lambda:1)
            self.cat_ls = [
                {"supercategory": "Licesne", "id": 1, "name": "LP"}
            ]
        else:
            # todo:
            pass

        aux_fn = existed_annotattion_file+'.aux'
        if os.path.exists(existed_annotattion_file) and os.path.exists(aux_fn):
            self.anno_dict = pickle.load(
                open(existed_annotattion_file,'rb')
            )
            self.aux_dict = pickle.load(open(aux_fn,'rb'))
        else:
            self.anno_dict = {
                "images": [],
                "annotations": [],
                "categories": self.categories,
            }
            self.aux_dict = EasyDict(
                img_prefix=img_prefix,
                next_image_id=0,
                next_anno_id=0,
                id_img_mapping=bidict()
            )
        self.check_path = check_path
        self.img_prefix = self.aux_dict['img_prefix']
    
    def add_dir(self, src_dir):
        file_ls = get_subfiles(src_dir)
        img_fn_ls = filter(
            lambda x: (x[x.rfind('.')+1:].lower() in img_post_ls) and (x[:len(self.img_prefix)] == self.img_prefix) , file_ls
        )
        relative_fn_ls = map(
            lambda x:x[len(self.img_prefix):],
            img_fn_ls
        )
        invers_dict = self.aux_dict['id_img_mapping'].inverse
        relative_fn_ls =list(relative_fn_ls)
        for i, fn in track(enumerate(relative_fn_ls), description=f'Processing {src_dir}',total=len(relative_fn_ls)):
            # add image into image info
            if fn in invers_dict.keys():
                pass
            else:
                full_img_fn = self.img_prefix + fn
                label_fn = makelabel_nm(full_img_fn)

                if (not os.path.exists(full_img_fn)) or (not os.path.exists(label_fn)):
                    continue

                cvt_res =None
                try:
                    cvt_res = self.cvt_annotation(full_img_fn, label_fn)
                except Exception as e:
                    print(f'failed in {full_img_fn} with {e}')
                    with open(self.check_path,'a+') as f:
                        f.write(f'{full_img_fn}\n')
                
                if cvt_res:
                    img_info, anno_info_ls = cvt_res
                    self.anno_dict['images'].append(img_info)

                    for img_anno_id, anno_dict in enumerate(anno_info_ls):
                        anno_dict['image_id'] = self.aux_dict['next_image_id'] + img_anno_id
                        anno_dict['id'] = self.aux_dict['next_anno_id'] + img_anno_id

                    self.anno_dict['annotations'] =self.anno_dict['annotations'] + anno_info_ls
                    self.aux_dict['id_img_mapping'][self.aux_dict['next_image_id']] = fn
                    self.aux_dict['next_image_id'] += 1
                    self.aux_dict['next_anno_id']+= len(anno_info_ls)

                else:
                    print(f'image: {full_img_fn} and label: {label_fn} passed')
        return
    
    def dump_res(self, annotation_file, if_keepold=True):
        aux_fn = annotation_file+'.aux'
        if if_keepold and annotation_file == self.existed_annotattion_file:
            anno_bkup_name = annotation_file + f'_{asctime()}'.replace(' ','').replace(':','')
            aux_bkup_name = anno_bkup_name + '.aux'
            shutil.copy(annotation_file,anno_bkup_name)
            shutil.copy(aux_fn, aux_bkup_name)
        with open(annotation_file, 'wb') as f:
            pickle.dump(self.anno_dict,f)
        with open(aux_fn, 'wb') as f:
            pickle.dump(self.aux_dict,f)
        return
    
    def export_to_annofile(self, out_dir,val_ratio=0.05,seed=None):

        if seed:
            r_seed = seed
        else:
            from random import seed,shuffle
            from time import time
            r_seed=time()
        seed(r_seed)
        img_ids = [x['id'] for x in self.anno_dict['images']]
        total_image_num = len(img_ids)
        total_img_ids = list(range(total_image_num))
        shuffle(total_img_ids)
        
        train_num = int(round((1- val_ratio) * total_image_num))
        val_num = total_image_num - train_num
        print(f'train image num:{train_num}\t, validataion image num:{val_num}')

        train_ids = [img_ids[i] for i in total_img_ids[:train_num]]
        val_ids = [img_ids[i] for i in total_img_ids[train_num:]]

        train_dict = {
                "images": [],
                "annotations": [],
                "categories":self.cat_ls,
            }
        val_dict = {
                "images": [],
                "annotations": [],
                "categories": self.cat_ls,
            }
        train_dict['images'] = [
            x for x in self.anno_dict['images'] if x['id'] in train_ids
        ]
        train_dict['annotations'] = [
            x for x in self.anno_dict['annotations'] if x['image_id'] in train_ids
        ]

        val_dict['images'] = [
            x for x in self.anno_dict['images'] if x['id'] in val_ids
        ]
        val_dict['annotations'] = [
            x for x in self.anno_dict['annotations'] if x['image_id'] in val_ids
        ]
        os.makedirs(out_dir, exist_ok=True)
        train_anno_file = os.path.join(out_dir,'train.json')
        val_anno_file = os.path.join(out_dir,'val.json')
        with open(train_anno_file,'w') as f:
            json.dump(train_dict,f)
        with open(val_anno_file,'w') as f:
            json.dump(val_dict,f)

        return
    
    def print_stastics_each_cls(self,):
        return

    
    def cvt_img_info(self, src_img_fn):
        img = safeimread(src_img_fn)
        if isinstance(img,np.ndarray):
            h,w,_ = img.shape
            img_info = {
            'id':self.aux_dict['next_image_id'],
            'width':w,
            'height':h,
            'file_name': src_img_fn[len(self.img_prefix):],
            }
            return img_info
        else:
            return None
        
    def cvt_anno_info(self, src_label_fn):
        if not os.path.exists(src_label_fn):
            return None
        anno_info_dict_ls = []
        lines = []
        with open(src_label_fn, 'r') as f:
            lines = f.readlines()

        for obi_in_img_id, line in enumerate(lines):
            obj_anno = json.loads(line.strip())
            anno_info_dict = {} 
            # {
            #     "category_id": lp_type_id,
            #     "segmentation":quad,
            #     "area": _quad.area,
            #     "bbox": quad_to_bbox(quad),
            #     "iscrowd": 0
            # }
            # 获取license id
            lp_type = ''

            if (obj_anno.get('Palate_visibleness','') == 'Unrecognizable'):
                lp_type = 'unrecognizable'
            elif('vehicle_type' in obj_anno.keys()):
                continue
            else:
                for candi_key in [
                'vehiclePlate',
                'license_type',
                'license_type',
                'vehiclePlate'
                ]: 
                    lp_type = obj_anno.get(candi_key,'')
                    if lp_type:
                        break
            
            if lp_type:
                # anno_info_dict['category_id'] = self.key_id_mapping[lp_type]
                if obj_anno.get('polygon', False):
                    if not obj_anno['polygon'].get('point3', False):
                        # valid_obj = False
                        raise( AssertionError(f'no point3 in {src_label_fn}'))
                    else:
                        quad = pt_tolist(obj_anno['polygon']['point0']) + pt_tolist(obj_anno['polygon']['point1']) + \
                        pt_tolist(obj_anno['polygon']['point2']) + \
                        pt_tolist(obj_anno['polygon']['point3'])
                elif obj_anno.get('rect', False):
                    quad = bbox_to_quad(pt_tolist(obj_anno['rect']))
                    pass
                else:
                    return None
                
                _quad = Polygon(
                    [   (quad[0], quad[1]),
                        (quad[2], quad[3]),
                        (quad[4], quad[5]),
                        (quad[6], quad[7])])
                
                anno_info_dict = {
                    # 'image_id':self.aux_dict['next_image_id'],
                    # "id":self.aux_dict['next_anno_id']+ obi_in_img_id,
                    "category_id": self.key_id_mapping[lp_type],
                    "segmentation":quad,
                    "area": _quad.area,
                    "bbox": quad_to_xywh(quad),
                    "iscrowd": 0
                }
                anno_info_dict_ls.append(anno_info_dict)
            else:
                return None
            
        return anno_info_dict_ls
            
    def cvt_annotation(self, src_img_fn, src_label_fn):

        rela_imgfn_index= self.aux_dict['id_img_mapping'].inverse

        register_imgfn = src_img_fn[len(self.img_prefix):]
        if (register_imgfn in rela_imgfn_index):
            return None

        img_info = self.cvt_img_info(src_img_fn)
        if img_info is None:
            return None

        anno_ls = self.cvt_anno_info(src_label_fn)
        
        if anno_ls:
            return img_info, anno_ls
        else:
            return None



    def get_data_stattics(self,AnnotationFile=None):
        out_dict = {
            k:v for k,v in self.type_counts_map.items()
        }
        return out_dict
    
    def type_statstics(self,):
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



if __name__ == '__main__':
    root_dir = '/media/112new_sde/LPD/DTC_RAW/'

    # chn_types_mapping = defaultdict()

    old_anno = '/media/112new_sde/LPD/COCOStyleAnnotations/LPD/OneClsIncludeUnrecognizable'
    # old_anno = ''

    cvt_tool = DTCRAWCvtTool(
        img_prefix=root_dir,
        existed_annotattion_file=old_anno,
        categories=['LP'],
        check_path='to_check.txt'
    )

    src_dir_ls = [
        '/media/112new_sde/LPD/DTC_RAW/Legacy/',
        '/media/112new_sde/LPD/DTC_RAW/DVR/',
        # '/media/112new_sde/LPD/DTC_RAW/BSD/',
        '/media/112new_sde/LPD/DTC_RAW/AVM/',
        '/media/112new_sde/LPD/DTC_RAW/APA/',
        '/media/112new_sde/LPD/DTC_RAW/HK_Macau/'
        # '/media/112new_sde/LPD/DTC_RAW/AVM/ArcSoft-AVM-L-22111DT12-T2-001/递交数据/长安AVM视频+图片'
    ]
    for pro_dir in src_dir_ls:
        print(f'processing {pro_dir}')
        cvt_tool.add_dir(pro_dir)

    cvt_tool.dump_res('/media/112new_sde/LPD/COCOStyleAnnotations/LPD/OneClsIncludeUnrecognizable')


    # anno_dir = '/media/112new_sde/LPD/COCOStyleAnnotations/LPD/onecls'
    anno_dir = '/media/112new_sde/LPD/COCOStyleAnnotations/LPD/onecls_all'
    val_ratio = 0.05
    cvt_tool.export_to_annofile(anno_dir,val_ratio)
    pass