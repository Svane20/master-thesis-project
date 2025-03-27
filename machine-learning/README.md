## Run training after SSH connection is closed 

1. Activate the thesis conda environment:

````bash
conda activate thesis
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

3. Check the running process

````bash
tail -f nohup.out
````

6. Kill the process with the following command:

````text
kill -9 <PID>
````