## Usage

 - ### First

`sh run_2steps.sh`

This step requires one GPU.

 - ### Second

`sh run_cpu_step.sh`

This step requires a big RAM (>350GB)

 - ### Third

`sh run_stitch_step.sh`

This step requires one GPU.

More specific, those `sh` files call following python programs.
```
python run_whole_brain/run_2steps.py <brain_tag> <pair_tag> <gpu_id>

python run_whole_brain/run_cpu_step.py <brain_tag> <pair_tag>

python run_whole_brain/run_stitch_step.py <brain_tag> <pair_tag> <gpu_id>
```