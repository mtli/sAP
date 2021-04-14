@echo off
:: For documentation, please refer to "doc\tasks.md"

set "dataDir=D:\Data"

set methodName=mrcnn50_nm
set scale=0.5

python forecast\streamer.py ^
	--data-root "%dataDir%\Argoverse-1.1\tracking" ^
	--annot-path "%dataDir%\Argoverse-HD\annotations\val.json" ^
	--fps 30 ^
	--eta 0 ^
	--config "..\mmdetection\configs\mask_rcnn\mask_rcnn_r50_fpn_2x_coco.py" ^
	--weights "%dataDir%\ModelZoo\mmdet\mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth" ^
	--in-scale 0.5 ^
	--no-mask ^
	--dynamic-schedule ^
	--runtime "%dataDir%\Exp\ArgoVerse1.1\runtime-zoo\1080ti\%methodName%_s%scale%.pkl" ^
	--out-dir "%dataDir%\Exp\ArgoVerse1.1\output\str_%methodName%_s%scale%\val" ^
	--overwrite ^
	&&
python det\streaming_eval.py ^
	--data-root "%dataDir%\Argoverse-1.1\tracking" ^
	--annot-path "%dataDir%\Argoverse-HD\annotations\val.json" ^
	--fps 30 ^
	--eta 0 ^
	--result-dir "%dataDir%\Exp\Argoverse-HD\output\str_%methodName%_s%scale%\val" ^
	--out-dir "%dataDir%\Exp\Argoverse-HD\output\str_%methodName%_s%scale%\val" ^
	--overwrite ^
	--vis-dir "%dataDir%\Exp\Argoverse-HD\vis\str_%methodName%_s%scale%\val" ^
	--vis-scale 0.5 ^
