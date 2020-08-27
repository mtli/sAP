@echo off
:: For documentation, please refer to "doc\tasks.md"

set "dataDir=D:\Data"

set methodName=mrcnn50_nm
set scale=0.5
set platform=1080ti

python util\add_to_runtime_zoo.py ^
	--time-info "%dataDir%\Exp\Argoverse-HD\output\rt_%methodName%_s%scale%\val\time_info.pkl" ^
	--out-path "%dataDir%\Exp\Argoverse-HD\runtime-zoo\%platform%\%methodName%_s%scale%.pkl" ^
	--overwrite ^
