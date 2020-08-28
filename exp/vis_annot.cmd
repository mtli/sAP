@echo off
:: For documentation, please refer to "doc\tasks.md"

set "dataDir=D:\Data"

python vis\vis_det_th.py ^
	--data-root "%dataDir%\Argoverse-1.1\tracking" ^
	--annot-path "%dataDir%\Argoverse-HD\annotations\val.json" ^
	--gt ^
	--vis-dir "$dataDir\Argoverse-HD\vis\val" ^
	--overwrite ^
