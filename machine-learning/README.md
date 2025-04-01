## Watch GPU during training

1. Run the following command:

## Linux

````bash
watch -n 1 nvidia-smi
````

## Windows

````bash
while ($true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 1 }
````

## Run training after SSH connection is closed 

1. Activate the thesis conda environment:

````bash
conda activate thesis
````

### DPT

1. Run this command from the root directory to run training

````bash
nohup python -m dpt.train &
````

2. Find the running process

````bash
ps -ef | grep dpt.train
````

### U-Net

1. Run this command from the root directory to run training

````bash
nohup python -m unet.train &
````

2. Find the running process

````bash
ps -ef | grep unet.train
````

### Common

2. Check the running process

````bash
tail -f nohup.out
````

3. Kill the process with the following command:

````text
kill -9 <PID>
````