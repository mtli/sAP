@echo off
:: For documentation, please refer to "doc\tasks.md"

set "dataDir=D:\Data"

set methodName=mrcnn50_nm
set scale=0.5

python forecast\pps_forecast_kf.py ^
	--data-root "%dataDir%\Argoverse-1.1\tracking" ^
	--annot-path "%dataDir%\Argoverse-HD\annotations\val.json" ^
	--fps 30 ^
	--eta 0 ^
	--assoc iou ^
	--forecast-before-assoc ^
	--in-dir "%dataDir%\Exp\Argoverse-HD\output\srt_%methodName%_ds_pf1.2_s%scale%\val" ^
	--out-dir "%dataDir%\Exp\Argoverse-HD\output\pps_%methodName%_ds_pf1.2_s%scale%_kf\val" ^
	--overwrite ^
	--vis-dir "%dataDir%\Exp\Argoverse-HD\vis\pps_%methodName%_ds_pf1.2_s%scale%_kf\val" ^
	--vis-scale 0.5 ^
