# Code Setup

## Requirement
You need an environment that has python and CUDA installed. For running on Windows, please read the additional notes [here](win_note.md).


## Installing dependencies

Most of the dependencies can be installed through this command with Conda environment. You might want to change the version for `cudatoolkit` in `environment.yml` to match your CUDA version <em>before</em> running it.

```
conda env create -f environment.yml
```

The created virtual environment is named `sap` and you can activate it by
```
conda activate sap
```

The next step is to install [mmdetection](https://github.com/open-mmlab/mmdetection) and its compatible version of mmcv. 

### mmcv installation

The command for installing `mmcv` is as follows:
```shell
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace `{cu_version}` and ``{torch_version}`` with the versions you are currently using.
You will get import or runtime errors if the versions are incorrect.

For example, with ``CUDA 10.2`` and ``PyTorch 1.6.0``, you can use the following command:

```shell
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
```
You should see that `pip` is downloading *a pre-compiled wheel file*:
```
Collecting mmcv-full==1.1.5
  Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/mmcv_full-1.1.5-cp38-cp38-manylinux1_x86_64.whl (18.5 MB)
```

If `pip` downloads a tar file:
```
Collecting mmcv-full==1.1.5
  Downloading mmcv-full-1.1.5.tar.gz (239 kB)
```
that means `mmcv` has not been compiled for your specific configuration.
We recommended you to change your CUDA or PyTorch versions.
Otherwise, you will need to compile `mmcv` from source to enable its CUDA components.

More information on `mmcv` installation can be found on their [Github page](https://github.com/open-mmlab/mmcv/).

### mmdetection installation

To install mmdetection, first clone the repo and checkout a specific version:
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout tags/v2.7.0
```

Then run:
```
pip install -v -e .  # or "python setup.py develop"
```

More information on `mmdetection` installation can be found their [Github page](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md).


## Prepare detection models

Download the pretrained model weights from mmdetection's [model zoo](https://github.com/open-mmlab/mmdetection/blob/3e902c3afc62693a71d672edab9b22e35f7d4776/docs/model_zoo.md). Please use the link above to access the right version of the model zoo to avoid any compatibility issues.

Note that Argoverse-HD is annotated according to COCO's format and class definitions. Therefore, it's reasonable to directly test out COCO pretrained models on Argoverse-HD.


## (Optionally) Compile the tracking association module
If you plan to use tracking or forecasting, you need to compile the IoU based association function. Change the directory back to this repo's root directory and run:
```
python setup.py build_ext --inplace
```


## Modify paths and run the scripts
The entry-point scripts for different tasks can be found under `exp/`. You need to modify the paths for the dataset, model configuration and weights, and the output folder before running them. <em>Note that those scripts should be run from the root directory of this repo</em>. For more information on these scripts, check out 

## Setup verification

If you have set it up correctly, running `exp/offline_det.sh` should be able to get you an AP of 21.8:
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.218
```
