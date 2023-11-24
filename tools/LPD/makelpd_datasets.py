import numpy as np
import mmcv
import pickle
import os
import sys
import cv2
import json
from PIL import Image
from collections import defaultdict, Mapping
from random import seed, shuffle
from tqdm import tqdm
import shutil
from time import asctime
import shapely
from shapely.geometry import Polygon


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


class PreparCocoStyleAnnotation:
    def __init__(
            self,
            valid_area_sz_range: list = [0.01, 0.8],
            ignore_type_ls: list = [],
            class_mapping=None,
            categories=['LP'],
            eval_ratio=0.05,
            img_post_ls=[
                'jpg',
                'png',
                'jpeg',
            ]
    ):
        self.valid_area_sz_range = valid_area_sz_range
        self.ignore_type_ls = ignore_type_ls
        self.class_mapping = class_mapping
        if not self.class_mapping:
            self.class_mapping = defaultdict(lambda: 1)
        self.categories = categories
        self.eval_ratio = eval_ratio
        self.img_post_ls = img_post_ls
        return

    def cvtInsToAnno(self, src_label_dict):
        lp_type = src_label_dict['vehiclePlate'] if src_label_dict.get(
            'vehiclePlate', False) else src_label_dict['license_type']

        lp_type_id = self.class_mapping[lp_type]

        valid_obj = True

        def quad_to_bbox(quad):
            bbox = [
                min(quad[::2]),
                min(quad[1::2]),
                max(quad[::2]),
                max(quad[1::2])]
            return bbox

        def quad_to_xywh(quad):
            bbox = [
                min(quad[::2]),
                min(quad[1::2]),
                max(quad[::2]),
                max(quad[1::2])]
            xywh = [
                bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3]-bbox[1]
            ]

            return xywh

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

        if src_label_dict.get('polygon', False):
            if not src_label_dict['polygon'].get('point3', False):
                valid_obj = False
            else:
                quad = pt_tolist(src_label_dict['polygon']['point0']) + pt_tolist(src_label_dict['polygon']['point1']) + \
                    pt_tolist(src_label_dict['polygon']['point2']) + \
                    pt_tolist(src_label_dict['polygon']['point3'])
        elif src_label_dict.get('rect', False):
            quad = bbox_to_quad(pt_tolist(src_label_dict['rect']))
            pass
        else:
            return None

        _quad = Polygon(
            [(quad[0], quad[1]),
             (quad[2], quad[3]),
             (quad[4], quad[5]),
             (quad[6], quad[7])])

        out_anno_dict = {
            "category_id": lp_type_id,
            "segmentation": quad,
            "area": _quad.area,
            "bbox": quad_to_bbox(quad),
            "iscrowd": 0}
        return out_anno_dict

    def process_raw_dtc_data(self, img_fn: str, label_fn: str, next_image_id: int, next_instance_id: int) -> list[dict, int, int]:

        srcimg = cv2.imread(img_fn, flags=cv2.IMREAD_COLOR +
                            cv2.IMREAD_IGNORE_ORIENTATION)
        if isinstance(srcimg, np.ndarray):
            pass
        else:
            raise AssertionError(f'{img_fn} can not be read')

        h, w, _ = srcimg.shape

        image_info_dict = {
            'id': next_image_id,
            'width': w,
            'height': h,
            'file_name': img_fn,
        }

        annotation_info_dict_ls = []

        lable_lines = []
        with open(label_fn, 'r') as f:
            lable_lines = f.readlines()
        # imageinfo instance_info
        for line in lable_lines:
            src_label_dict = json.loads(line.strip())
            cvtted_anno_dict = self.cvtInsToAnno(src_label_dict)

            if cvtted_anno_dict:
                anno_base = {
                    'id': next_instance_id,
                    'image_id': next_image_id,
                }
                cvtted_anno_dict.update(anno_base)
                next_instance_id += 1

                annotation_info_dict_ls.append(cvtted_anno_dict)
        return image_info_dict, annotation_info_dict_ls

    def init_anno(self,):
        anno_dict = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        for i, cls in enumerate(self.categories):
            anno_dict["categories"].append(
                {
                    'supercategory': 'Licesne',
                    'id': i+1,
                    'name': cls
                }
            )

        return anno_dict

    def process_pair_ls(self, pair_ls, anno_dict, next_image_id=0, next_instance_id=0):
        for i, fn_pair in tqdm(enumerate(pair_ls)):
            #
            img_fn, label_fn = fn_pair
            parsed_data = self.process_raw_dtc_data(
                img_fn,
                label_fn,
                next_image_id,
                next_instance_id)
            if parsed_data:
                image_info_dict, anno_info_dict_ls = parsed_data
                anno_dict['images'].append(image_info_dict)
                anno_dict['annotations'] = anno_dict['annotations'] + \
                    anno_info_dict_ls
                next_image_id += 1
                next_instance_id += len(anno_info_dict_ls)
            else:
                print(f"jumping over {img_fn}")
                continue

        return

    # process multiple dir
    def process_dirs(self, src_dir_ls: list, export_dir: str, ifMergeOriginal=False):

        train_anno_dict = self.init_anno()
        val_anno_dict = self.init_anno()
        if ifMergeOriginal:
            # TODO: add merge new dirs
            pass

        pair_ls = []
        for src_dir in src_dir_ls:
            all_files = get_subfiles(src_dir)
            img_files = filter(
                lambda x: x[x.rfind('.')+1:].lower() in self.img_post_ls,
                all_files)
            pair_ls = pair_ls + \
                list(map(lambda x: (x, x[:x.rfind('.')] + '.txt'), img_files))

        shuffle(pair_ls)
        total_instance_num = len(pair_ls)
        eval_num = int(round(total_instance_num * self.eval_ratio))
        train_num = total_instance_num - eval_num

        train_ls = pair_ls[:train_num]
        train_ls.sort()
        eval_ls = pair_ls[-eval_num:]
        eval_ls.sort()
        print('processing train data')
        self.process_pair_ls(train_ls, train_anno_dict, 0, 0)
        print('processing validation data')
        self.process_pair_ls(eval_ls, val_anno_dict, 0, 0)
        # export data

        os.makedirs(export_dir, exist_ok=True)

        out_train_fn = os.path.join(export_dir, f'instances_train.json')
        out_val_fn = os.path.join(export_dir, f'instances_val.json')
        json.dump(train_anno_dict, open(out_train_fn, 'w'))
        json.dump(val_anno_dict, open(out_val_fn, 'w'))
        return


if __name__ == '__main__':

    from time import time
    import argparse
    seed(time())
    cvt = PreparCocoStyleAnnotation(
        valid_area_sz_range=[0.01, 0.8],
        ignore_type_ls=[],
        class_mapping=None,
        categories=['LP'],
        eval_ratio=0.05,
        img_post_ls=[
            'jpg',
            'png',
            'jpeg',
        ]
    )

    src_dir_ls = [
        '/media/112new_sde/LPD/DTC_RAW/BSD'
    ]
    export_dir = '/media/112new_sde/LPD/COCOStyleAnnotations/BSD_ONLY'

    cvt.process_dirs(src_dir_ls, export_dir)
