# For documentation, please refer to "doc/tasks.md"

dataDir="/data2/mengtial"

methodName=mrcnn50_nm
scale=0.5

python det/rt_det.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--config "$HOME/repo/mmdetection-sap/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py" \
	--weights "$dataDir/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth" \
	--in-scale $scale \
	--no-mask \
	--out-dir "$dataDir/Exp/Argoverse-HD/output/rt_${methodName}_s${scale}/val" \
	--overwrite \
    && 
python det/streaming_eval.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--eta 0 \
	--result-dir "$dataDir/Exp/Argoverse-HD/output/rt_${methodName}_s${scale}/val" \
	--out-dir "$dataDir/Exp/Argoverse-HD/output/rt_${methodName}_s${scale}/val" \
	--overwrite \
	--vis-dir "$dataDir/Exp/Argoverse-HD/vis/rt_${methodName}_s${scale}/val" \
	--vis-scale 0.5 \
