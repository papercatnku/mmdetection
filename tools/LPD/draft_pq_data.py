import os
import os.path as osp
import numpy as np
from shutil import copyfile
'''
for generating postquant data
'''
def get_subfiles(path, length=None):
    out_ls = []

    def _func(x, ls=[]):
        for fn in os.listdir(x):
            full_path = os.path.join(x, fn)
            if os.path.isdir(full_path):
                _func(full_path, ls)
            else:
                ls.append(full_path)
            if length!=None and len(out_ls)>length:
                return
        return
    _func(path, out_ls)
    return out_ls


def random_draft_image(src_dir, dst_dir, num=100):
    src_fns = get_subfiles(src_dir)
    src_fns = [fn for fn in src_fns if osp.splitext(fn)[-1].lower() in ['.jpg', '.png', '.jpeg']]
    draft = np.random.choice(src_fns, num, replace=False)
    os.makedirs(dst_dir, exist_ok=True)
    for fn in draft:
        osp.basename(fn)
        dst_fn =osp.join(dst_dir, osp.basename(fn))
        copyfile(fn, dst_fn)
    return


if __name__ == '__main__':
    src_dir = '/media/112new_sde/LPD/DTC_RAW/AVM_SENTRY'
    dst_dir = '/media/21sdg/zcy6735/misc/pd_data/AVM_SENTRY'
    draft_num = 50

    random_draft_image(src_dir, dst_dir, draft_num)



