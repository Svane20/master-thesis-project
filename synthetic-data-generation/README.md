# Blender

## Libraries

### pydelatin

1. Install the library:

````bash
conda install -c conda-forge pydelatin
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
nohup python main.py &
````

4. When the SSH connection is re-established, run the following command:

````bash
ps -ef | grep main.py
````

OR

5. Check the logs in the `logs/blender-run.log` file with the following command:

Current run

````bash
tail -f logs/blender-run.log
````

Current iteration

````bash
tail -f logs/blender-app.log
````

6. Kill the process with the following command:

````bash
kill -9 <PID>
````

7. Run the following command to check if the process is killed:

````bash
ps -ef | grep main.py
````

## Clear memory Linux buff/cache

1. Run the following command:

````bash
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
````