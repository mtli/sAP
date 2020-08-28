# For documentation, please refer to "doc/tasks.md"

dataDir="/data2/mengtial"

python vis/vis_contrast.py \
	--dir-A "${dataDir}/Exp/Argoverse-HD/vis-th0.5/rt_mrcnn50_nm_s0.5/val" \
	--dir-B "${dataDir}/Exp/Argoverse-HD/vis-th0.5/srt_mrcnn50_nm_inf_s0.5/val" \
	--seq b1ca08f1-24b0-3c39-ba4e-d5a92868462c \
	--horizontal \
	--split-pos 0.55 \
	--split-animation swing \
	--out-dir "${dataDir}/Exp/Argoverse-HD/vis-th0.5/single-vs-inf/val" \
	--make-video \
	--overwrite \
