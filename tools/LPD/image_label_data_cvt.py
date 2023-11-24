import numpy as np
import mmcv
import pickle
import os
import sys
import cv2
import json
from PIL import Image
from collections import defaultdict
from random import seed, shuffle
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
import multiprocessing
import threading
from multiprocessing import Process, Pool, Lock, Manager
from random import shuffle, seed
from time import time
from tqdm import tqdm


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


img_post_ls = ['jpg', 'png', 'jpeg',]


def makelabel_nm(x):
    return x[:x.rfind('.')] + '.txt'


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


# class parse_raw
lock = Lock()


class mulpti_process_base:
    def __init__(self, base_dir='', num_process=8, illegal_log_fn=f'./tocheck-{asctime()}.txt'):
        seed(time())
        self.num_process = num_process
        # self.lock = Lock()
        self.illegal_log_fn = illegal_log_fn
        self.mem_ls = []
        self.base_dir = base_dir
        self.train_ls, self.val_ls = [], []

        return

    def split_train_val(self, val_ratio=0.05, debug_num=None):
        shuffle(self.mem_ls)
        total_num = len(self.mem_ls)
        if debug_num:
            total_num = debug_num
        val_num = int(round(val_ratio*total_num))
        train_num = total_num - val_num

        self.train_ls, self.val_ls = self.mem_ls[:
                                                 train_num], self.mem_ls[train_num:total_num]
        print(f'train num: {train_num}\tval num: {val_num}')
        return

    def process_dir(self, src_dir, *args, **kwargs):
        raw_tuple_ls = self.get_raw_list(src_dir)
        # pbar = tqdm(total=len(raw_tuple_ls))
        # def update_bar_wrapper(x):
        #     res = self.parse_raw_annotation(x)
        #     pbar.update(1)
        #     return res
        pp = Pool(self.num_process)
        total_out_ls = pp.map(
            self.parse_raw_annotation,
            raw_tuple_ls
        )
        pp.close()
        pp.join()
        self.mem_ls = self.mem_ls + \
            list(filter(lambda x: not (x is None), total_out_ls))
        print(f'{len(self.mem_ls)} pairs from {src_dir} added')
        return

    def get_raw_list(self, src_dir):
        src_fn_ls = get_subfiles(src_dir)
        img_fn_list = list(filter(
            lambda x: (x[x.rfind('.')+1:].lower() in img_post_ls),
            src_fn_ls
        ))
        pass
        img_label_tuple_ls = [
            (img_fn, makelabel_nm(img_fn)) for img_fn in track(img_fn_list, description=f'fetching all pairs in {src_dir}') if (os.path.exists(img_fn) and os.path.exists(makelabel_nm(img_fn)))]
        return img_label_tuple_ls

    def split_task(self, total_list):
        total_list_num = len(total_list)
        each_part_num = total_list_num // self.num_process
        out_ls = []
        for i in range(self.num_process-1):
            out_ls.append(total_list[i * each_part_num: (i+1) * each_part_num])
        out_ls.append(total_list[(self.num_process-1) * each_part_num:])
        return out_ls

    def parse_raw_annotation(self, data):
        try:
            img_fn, label_fn = data
            img_info = self.parse_img_info(img_fn)
            label_info = self.parse_label_info(label_fn)

        except Exception as e:
            # with self.lock:
            print(e)
            with lock:
                with open(self.illegal_log_fn, 'a+') as f:
                    f.write(f'{img_fn}\n')
                    f.write(f'{label_fn}\n')
            return None

        if img_info and label_info:
            return img_info, label_info
        return None

    def parse_img_info(self, img_fn):
        srcimg = safeimread(img_fn)
        if not isinstance(srcimg, np.ndarray):
            raise AssertionError(f'failed to read {img_fn}')

        h, w, c = srcimg.shape
        out_dict = {
            'file_name': os.path.relpath(img_fn, self.base_dir),
            'width': w,
            'height': h,
        }
        return out_dict

    def parse_label_info(self, label_fn):
        return {}

    def export(self, out_dir, type='COCO'):
        return


class obj_det_cvt_base(mulpti_process_base):
    def __init__(self, base_dir='', num_process=8, illegal_log_fn=f'./tocheck-{asctime()}.txt'):
        super().__init__(base_dir, num_process, illegal_log_fn)
        # customize
        self.init_type_config()
        return

    def init_type_config(self):
        self.rawtype_innertype_mapping = {}
        self.innertype_id_mapping = {'Obj': (0, 'Obj')}
        return

    def parse_label_info(self, label_fn):
        out_dict = {
            'quad_ls': []
        }
        lines = []
        with open(label_fn, 'r') as f:
            lines = f.readlines()

        for line in lines:
            ins_dict = json.loads(line.strip())
            lp_type = ''
            if ('vehicle_type' in ins_dict.keys()):
                # legacy中是车辆相关标注的部分
                continue
            elif (ins_dict.get('Palate_visibleness', '') == 'Unrecognizable'):
                lp_type = 'unrecognizable'
            else:
                for candi_key in [
                    'vehiclePlate',
                    'license_type',
                    'license_type',
                        'vehiclePlate']:
                    lp_type = ins_dict.get(candi_key, '')
                    if lp_type:
                        break
            if lp_type:
                if ins_dict.get('polygon', False):
                    if not ins_dict['polygon'].get('point3', False):
                        # valid_obj = False
                        raise (AssertionError(f'no point3 in {label_fn}'))
                    else:
                        quad = pt_tolist(ins_dict['polygon']['point0']) + pt_tolist(ins_dict['polygon']['point1']) + pt_tolist(
                            ins_dict['polygon']['point2']) + pt_tolist(ins_dict['polygon']['point3'])
                elif ins_dict.get('rect', False):
                    quad = bbox_to_quad(pt_tolist(ins_dict['rect']))
                else:
                    raise (AssertionError(
                        f'no valid quad annotation in {label_fn}'))

                quad_dict = {
                    'type': lp_type,
                    'quad': quad}

                out_dict['quad_ls'].append(quad_dict)
            else:
                continue
        return out_dict

    def export(self, out_dir, type='COCO'):
        os.makedirs(out_dir, exist_ok=True)
        if type == 'COCO':
            self.export_coco(out_dir)
        elif type == 'VOC':
            pass
        elif type == 'CUSTOM_PKL':
            pass
        else:
            return

    def if_label_valid(self, quad_info):
        return True

    def filt_quad_label(self, img_info, label_info):
        out_quad_info_ls = []
        out_labelinfo = label_info.copy()
        for quad in label_info['quad_ls']:
            if (self.if_label_valid(quad)):
                out_quad_info_ls.append(quad.copy())

        out_labelinfo['quad_ls'] = out_quad_info_ls

        return img_info, out_labelinfo

    def export_coco(self, out_dir):
        # train_anno_file
        train_anno_file = os.path.join(out_dir, 'train_annotations.json')
        val_anno_file = os.path.join(out_dir, 'val_annotations.json')

        cats = self.init_coco_cat()
        coco_train_dict, coco_val_dict = {
            "categories": cats}, {"categories": cats}

        next_image_id = 0
        next_anno_id = 0

        train_img_info_ls,  train_anno_info_ls, next_image_id, next_anno_id = self.make_coco_from_ls(
            self.train_ls, '-train', next_image_id, next_anno_id)
        val_img_info_ls,  val_anno_info_ls, next_image_id, next_anno_id = self.make_coco_from_ls(
            self.val_ls, '-train', next_image_id, next_anno_id)

        print(
            f'total image num: {next_image_id}\t annotation num:{next_anno_id}')

        print(
            f'train image num: {len(train_img_info_ls)}\t annotation num:{len(train_anno_info_ls)}')
        print(
            f'val image num: {len(val_img_info_ls)}\t annotation num:{len(val_anno_info_ls)}')

        coco_train_dict['images'] = train_img_info_ls
        coco_train_dict['annotations'] = train_anno_info_ls
        coco_val_dict['images'] = val_img_info_ls
        coco_val_dict['annotations'] = val_anno_info_ls

        with open(train_anno_file, 'w') as f:
            json.dump(coco_train_dict, f, ensure_ascii=False)

        with open(val_anno_file, 'w') as f:
            json.dump(coco_val_dict, f, ensure_ascii=False)
        return

    def init_coco_cat(self):
        cats = []
        for k, v in self.innertype_id_mapping.items():
            cls_id, super_type = v
            cats.append(
                {
                    "supercategory": super_type, "id": cls_id+1, "name": k,
                }
            )
        return cats

    def make_coco_from_ls(self, pair_ls, description_tag='', next_image_id=0, next_anno_id=0):
        img_info_ls = []
        anno_info_ls = []
        for pair in track(pair_ls, description=f'makecoco{description_tag}', total=len(pair_ls)):
            img_info, label_info = pair
            filted_img_info, filted_label_infp = self.filt_quad_label(
                img_info, label_info)
            if filted_img_info:
                coco_img_inf = {
                    'id': next_image_id,
                    'width': filted_img_info['width'],
                    'height': filted_img_info['height'],
                    'file_name': filted_img_info['file_name'],
                }
                if filted_label_infp:
                    for quad_ins in filted_label_infp['quad_ls']:
                        quad = quad_ins['quad']
                        _quad = Polygon(
                            [
                                (quad[0], quad[1]),
                                (quad[2], quad[3]),
                                (quad[4], quad[5]),
                                (quad[6], quad[7])])
                        coco_anno_dict = {
                            'id': next_anno_id,
                            'image_id': next_image_id,
                            'category_id': 1 + self.innertype_id_mapping[self.rawtype_innertype_mapping[quad_ins['type']]][0],
                            'segmentation': quad,
                            'area': _quad.area,
                            'bbox': quad_to_xywh(quad),
                            'iscrowd': 0
                        }
                        anno_info_ls.append(coco_anno_dict)
                        next_anno_id += 1
                img_info_ls.append(coco_img_inf)
                next_image_id += 1
        return img_info_ls, anno_info_ls, next_image_id, next_anno_id


class lpd_one_cls_cvt(obj_det_cvt_base):
    def init_type_config(self):
        # self.rawtype_innertype_mapping = defaultdict(lambda: 'LP')
        self.rawtype_innertype_mapping = {
            'blue_Plate': 'LP',
            'large_green_Plate': 'LP',
            'unrecognizable': 'LP',
            'other_Plate': 'LP',
            'yellow-black-single_Plate': 'LP',
            'motorcycle_Plate': 'LP',
            'yellow-black-double_Plate': 'LP',
            # 'paint_Plate': 'LP',
            'green_Plate': 'LP',
            'yellow_black_single_Plate': 'LP',
            'white-black_Plate': 'LP',
            'yellow_black_double_Plate': 'LP',
            'white_Plate': 'LP',
            'black-white_Plate': 'LP',
            'electric_Motorcycle_Plate': 'LP',
            'military_Plate': 'LP',
            'black_Plate': 'LP',
            'police_Plate': 'LP'
        }
        self.ignore_type_ls = [
            'paint_Plate'
        ]
        # innertype => superid, id
        self.innertype_id_mapping = {
            'LP': (0, 'LP')
        }

        return

    def if_label_valid(self, quad_info):

        if quad_info['type'] in self.ignore_type_ls:
            return False
        return True


class lpd_multi_cls_cvt(obj_det_cvt_base):
    # TODO: 修改适配多类别检测需要
    def init_type_config(self):
        self.rawtype_innertype_mapping = {
            'blue_Plate': 'BLUE',
            'large_green_Plate': 'GREEN',
            'unrecognizable': 'UNREC',
            'other_Plate': 'OTHER',
            'yellow-black-single_Plate': 'YELLOW_1',
            'motorcycle_Plate': 'OTHER',
            'yellow-black-double_Plate': 'YELLOW_2',
            'paint_Plate': 'PAINT',
            'green_Plate': 'GREEN',
            'yellow_black_single_Plate': 'YELLOW_1',
            'white-black_Plate': 'WHITE',
            'yellow_black_double_Plate': 'YELLOW_2',
            'white_Plate': 'WHITE',
            'black-white_Plate': 'BLACK',
            'electric_Motorcycle_Plate': 'OTHER',
            'military_Plate': 'OTHER',
            'black_Plate': 'BLACK',
            'police_Plate': 'WHITE'
        }
        # innertype => superid, id
        self.innertype_id_mapping = {
            'BLUE':     (0, 'COMMON'),
            'GREEN':    (1, 'COMMON'),
            'YELLOW_1': (2, 'COMMON'),
            'YELLOW_2': (3, 'COMMON'),
            'WHITE':    (4, 'UNCOMMON'),
            'BLACK':    (5, 'UNCOMMON'),
            'PAINT':    (6, 'UNCOMMON'),
            'OTHER':    (7, 'OTHER'),
            'UNREC':    (8, 'UNREC'),
        }

        return


if __name__ == '__main__':

    base_dir = '/media/112new_sde/LPD/DTC_RAW/'
    src_dir_ls = [
        '/media/112new_sde/LPD/DTC_RAW/Legacy/',
        '/media/112new_sde/LPD/DTC_RAW/DVR/',
        '/media/112new_sde/LPD/DTC_RAW/BSD/',
        '/media/112new_sde/LPD/DTC_RAW/AVM/',
        '/media/112new_sde/LPD/DTC_RAW/APA/',
        '/media/112new_sde/LPD/DTC_RAW/HK_Macau/',
        '/media/112new_sde/LPD/DTC_RAW/AVM_SENTRY/'
    ]
    val_ratio = 0.05
    # out_coco = '/media/112new_sde/LPD/LPDAnnotations/coco_style/cls_1_nopaint/'
    out_coco = '/media/112new_sde/LPD/LPDAnnotations/coco_style/cls_1_nopaint_231115/'
    cvt = lpd_one_cls_cvt(
        base_dir=base_dir,
        num_process=20,
    )
    # out_coco = '/media/112new_sde/LPD/LPDAnnotations/coco_style/cls_8'
    # cvt = lpd_multi_cls_cvt(
    #     base_dir=base_dir,
    #     num_process=20,
    # )
    for cur_dir in src_dir_ls:
        cvt.process_dir(cur_dir)
    # cvt.split_train_val(val_ratio=val_ratio,debug_num=4000)
    cvt.split_train_val(val_ratio=val_ratio)
    cvt.export(out_coco, type='COCO')
    pass
