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

1. CD to the project directory:

````text
cd master-thesis-project/synthetic-data-generation/
````

2. Activate the blender conda environment:

````bash
conda activate blender
````

3. Run the following command:

````bash
nohup python run.py &
````

4. When the SSH connection is re-established, run the following command:

````bash
ps -ef | grep main.py
````

OR 

5. Check the logs in the `logs/run.log` file with the following command:

````bash
tail -f logs/run.log
````

6. Kill the process with the following command:

````bash
kill -9 <PID>
````