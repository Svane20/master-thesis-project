# Blender

## Libraries

### pydelatin

1. Install the library:

````bash
conda install -c conda-forge pydelatin
````

## Run Blender UI with script execution

1. Run the following command:

````bash
python main.py
````

## Running multiple runs after SSH connection is closed

1. Run the following command:

````bash
nohup python main.py &
````

2. When the SSH connection is re-established, run the following command:

````bash
ps -ef | grep main.py
````

OR 

3. Check the logs in the `logs/run.log` file with the following command:

````bash
tail -f logs/run.log
````