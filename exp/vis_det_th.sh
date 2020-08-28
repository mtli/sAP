# For documentation, please refer to "doc/tasks.md"

dataDir="/data2/mengtial"

methodName=rt_mrcnn50_nm_s0.5

python vis/vis_det_th.py \
	--data-root "${dataDir}/Argoverse-1.1/tracking" \
	--annot-path "${dataDir}/Argoverse-HD/annotations/val.json" \
	--result-path "${dataDir}/Exp/Argoverse-HD/output/${methodName}/val/results_ccf.pkl" \
	--vis-dir "${dataDir}/Exp/Argoverse-HD/vis-th0.5/${methodName}/val" \
	--overwrite \
    --score-th 0.5 \
