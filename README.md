# sAP &mdash; Code for Towards Streaming Perception

<p align="center"><img alt="Teaser" src="doc/img/streaming.jpg" width="600px"></p>

This repo contains code for our ECCV 2020 [**paper**](https://www.cs.cmu.edu/~mengtial/proj/streaming/Martin%20-%20Streaming%20Perception.pdf?v=0.1) (Towards Streaming Perception). sAP stands for streaming Average Precision.

The dataset used in this project (Argoverse-HD) can be found on the [**project page**](http://www.cs.cmu.edu/~mengtial/proj/streaming/).


## Contents

- Offline detection
- Streaming (real-time online) detection
- Streaming tracking \& forecasting (Coming soon!)
- Simulated streaming detection, tracking, \& forecasting (Coming soon!)
- Simulated streaming detection, tracking, \& forecasting with infinite GPUs (Coming soon!)
- Meta-detector Streamer (Coming soon!)
- Streaming evaluation
- Single-frame schedule simulator
- Helper functions for visualization (Coming soon!)


## Getting started

1. Follow the instructions [here](doc/data_setup.md) to download and setup the dataset.
1. Follow the instructions [here](doc/code_setup.md) to install the dependencies.
1. Check out the examples to run different tasks in `exp/*`.


## Citation
If you use the code or the data for your research, please cite the paper:

```
@article{Li2020StreamingP,
  title={Towards Streaming Perception},
  author={Li, Mengtian and Wang, Yuxiong and Ramanan, Deva},
  journal={ECCV},
  year={2020}
}
```

## Acknowledgement
We would like to thank the [mmdetection](https://github.com/open-mmlab/mmdetection) team for implementing so many different detectors in a single awesome repo with a unified interface! This greatly reduced our efforts to evaluate different detectors under our streaming setting.
