# Additional Notes For Running on a Windows Machine

The support for running on Windows is experimental since we mainly tested on Linux machines. In additional to the [code setup instructions](code_setup.md), you also need to meet some hardware requirement and install some dependencies differently.

## Hardware requirement

While this code runs with any modern desktop GPU in Windows, latency measures might be off and do not reflect the full capability of the hardware. By default, GTX/RTX GPUs operate in WDDM mode (intended for graphics use) instead of TCC mode (intended for general-purpose computation). Latency is not optimized for computation tasks under WDDM and you might observe runtime much larger than what's reported in the paper. To avoid such issues, you need to operate in TCC mode. Measured runtime in TCC mode in Windows should match that in Linux. However, due to NVIDIA's restriction, only certain GPUs can be put into TCC mode in Windows. Commonly known models that allow TCC mode in Windows are TITAN RTX and all Tesla series including V100. In summary, if you want to obtain reasonable results for latency-sensitive parts of this codebase in Windows, you need one of the above GPUs.

## Installing dependencies

Since the old mmdetection doesn't support Windows, you will need to switch to a newer version and with a newer mmcv.

First, set up the conda environment in the same way as in the [code setup instructions](code_setup.md):
```
conda env create -f environment.yml
conda activate sap
```

Next, install mmcv:
```
pip install mmcv==0.2.15
```

Clone mmdetection and switch to <em>a different version</em>
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout e60d34af8b0ec8f6e6439a0f3fb687042e9dde82
```

The last step is the same (install within the `sap` environment):
```
pip install -e .
```


## Running the scripts
Instead of running the `*.sh` scripts, you run `*.cmd` counterparts in Windows.
