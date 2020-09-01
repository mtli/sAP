# Tasks
This doc describes tasks supported by this repo. Note that some concepts might require reading the paper or watching the talk on the [project page](http://www.cs.cmu.edu/~mengtial/proj/streaming/) to understand.


## Offline Detection
This is the standard task of running a detector on a dataset, and can be used as a sanity check to see if the data and code are correctly set up. The code outputs the detection output for each frame and optionally visualizes them if provided with a visualization folder (`--vis-dir`). By default, the code also evaluates the output using `COCOeval` (our Argoverse-HD is provided in COCO format and thus it works with COCO evaluation tools). 

Code: `exp/offline_det.sh`, `det/det_offline.py`


## Streaming Detection 

In this task, the detector processes image frames from a sensor stream in a real-time online fashion. Frames may be skipped if the detector fails to catch up (as our analysis suggests: <em>it's OK to skip frames!</em>). The sensor stream is simulated by playing back videos (or extracted image frames from videos) in real-time. Before the real-time playback, all frames in the video are loaded to system memory so that disk I/O does not introduce any latency. The output of streaming detection is timestamped output.

Code: `exp/sim_streaming_det.sh`, `det/rt_det.py`


## Streaming Evaluation 

Streaming evaluation takes input timestamped detections and associates that with the corresponding image frames (time-based alignment) and then evaluates the prediction based on this correspondence. Optionally, it visualizes the detection results.

Code: `exp/sim_streaming_det.sh`, `det/streaming_eval.py`


## Creating Runtime Profiles
We need runtime distributions to simulate streaming detection (explained in the next section) and an easy to create such a runtime profile (a collection of runtime distributions of all independent modules, e.g., detection, tracking and forecasting) is extracting runtime samples from the previous experiment runs (forming an empirical distribution). Also, this code assumes all runtime profiles are stored in a runtime zoo (analogous to a model zoo). After you have run streaming detection at least once, the runtime samples will be collected automatically and you can add them to the runtime zoo using `util/add_to_runtime_zoo.py`.

Code: `exp/extract_runtime.sh`, `util/add_to_runtime_zoo.py`


## Simulated Streaming Detection

This task achieves the same goal as streaming detection except that both the detection output and the runtime can be simulated. There are two levels of simulation. The first level only simulates runtime. In this case, the code does not measure the runtime of algorithms but draws a sample from the provided runtime distribution (see the previous section for runtime distributions). The second level simulates both the runtime and the detection results. In this case, in addition to providing runtime distributions, you also need to provide per-frame detection results on the entire dataset. The simulator will fetch the corresponding result based on the sampled runtime. The output format is the same as streaming detection. Note that runtimes can be scaled during simulation using argument `--perf-factor` (performance factor, e.g., 1.2 means 20% faster).

<p align="center"><img alt="computational constraints" src="img/fig4.png"></p>

By default, the code assumes a single GPU model, while code with `_inf` suffix assumes an infinite GPU model.

Code: `exp/sim_streaming_det.sh`, `det/srt_det.py`, `exp/sim_streaming_det_inf.sh`, `det/srt_det_inf.py`


## (More is coming soon)


