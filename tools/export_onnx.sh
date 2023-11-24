${CFG_PATH}="/media/112new_sde/ModelZoo/lpd/lpd_1cls_fpn_384_lite/lpd_1cls_384_fpn_forrv1106.py"
${PTH_PATH}="/media/112new_sde/ModelZoo/lpd/lpd_1cls_fpn_384_lite/best_bbox_mAP_epoch_300.pth"
${OUT_NAMES}="pred_cls_0,pred_cls_1,bbox_cls_0,bbox_cls_1,bbox_cls_0,bbox_cls_1"

${W}="384"
${H}="256"
${ONNX_PATH}="/media/112new_sde/ModelZoo/lpd/lpd_1cls_fpn_384_lite/lpd_1cls_384_fpn_forrv1106_${W}x{H}.onnx"


CUDA_VISIBLE_DEVICES=-1 python tools/deployment/pytorch2onnx.py ${CFG_PATH} ${PTH_PATH} --output-file ${ONNX_PATH} --shape "${H} ${W}" --output-names ${OUT_NAMES} --skip-postprocess --simplify