# Data Setup

## Requirement
You need an environment that has python and CUDA installed.


## Installing dependencies

Most of the dependencies can be installed through this command with Conda environment. You might want to change the version for `cudatoolkit` in `environment.yml` to match your CUDA version <em>before</em> running it.

```
conda env create -f environment.yml
```

The created virtual environment is named `sap` and you can activate it by
```
conda activate sap
```

The next step is to install [mmdetection](https://github.com/open-mmlab/mmdetection) and its compatiable version of mmcv. First, install mmcv:
```
pip install mmcv==0.2.11
```

Then clone mmdetection and switch to a specific version. Pick a suitable location (not within this repo) for cloning, and take note of it since you will later need to refer to its model configurations.
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 36b6577e3458c0e068f4523d60ce8c6e7c19febf
```

Install mmdetection within the `sap` environment:
```
pip install -e .
```
This step will compile some CUDA and C++ files and might take some time.


## Prepare detection models

Download the pretrained model weights from mmdetection's [model zoo](https://github.com/open-mmlab/mmdetection/blob/36b6577e3458c0e068f4523d60ce8c6e7c19febf/MODEL_ZOO.md). Please use the link above to access the right version of the model zoo to avoid any compatibility issues.

Note that Argoverse-HD is annotated according to COCO's format and class definitions. Therefore, it's reasonable to directly test out COCO pretrained models on Argoverse-HD.


## (Optionally) Compile the tracking association module
If you plan to use tracking or forecasting, you need to compile the IoU based association function. Change the directory back to this repo's root directory and run:
```
python setup.py build
```


## Modify paths and run the scripts
The entry-point scripts for different tasks can be found under `exp/`. You need to modify the paths for the dataset, model configuration and weights, and the output folder before running them. <em>Note that those scripts should be run from the root directory of this repo</em>.
