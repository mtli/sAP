# For documentation, please refer to "doc/tasks.md"

dataDir="/data2/mengtial"

methodName=mrcnn50_nm
scale=0.5
platform=1080ti

python util/add_to_runtime_zoo.py \
	--time-info "$dataDir/Exp/Argoverse-HD/output/rt_${methodName}_s${scale}/val/time_info.pkl" \
	--out-path "$dataDir/Exp/Argoverse-HD/runtime-zoo/$platform/${methodName}_s${scale}.pkl" \
	--overwrite \
