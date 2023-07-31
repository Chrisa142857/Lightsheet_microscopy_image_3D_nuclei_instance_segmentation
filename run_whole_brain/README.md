## Usage
You can get an example on `run_whole_brain/run_pipeline.sh`

More specific:

### Step 1
This step is using GPU.
```
cd ..
python run_whole_brain/run_2steps.py <brain_tag> <pair_tag> <gpu_id>
```
### Step 2
This step is using CPU.
```
python run_whole_brain/run_cpu_step.py <brain_tag> <pair_tag>
```
### Step 3
This step is using GPU.
```
python run_whole_brain/run_stitch_step.py <brain_tag> <pair_tag>
```