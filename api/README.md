# master-thesis-project

## Run Blender Python API as Standalone Module 

### Windows

1. Add Blender to Environment Variables under `System Variables` with the path to the Blender executable.

````text
C:\Program Files\Blender Foundation\Blender X.X\
````

2. Restart the PC to apply the changes.

3. Set up the Project Interpreter to use Python 3.11 

4. Install the required dependencies

````bash
pip install -r requirements.txt
````

## Run FAST API - DEV

1. Run the following command to start the server

````text
fastapi dev main.py
````

2. Open the browser and navigate to [localhost:8000/docs](http://localhost:8000/docs) to access the API documentation.