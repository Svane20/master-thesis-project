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
ps -ef | grep run.py
````

OR

5. Check the logs in the `logs/run.log` file with the following command:

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

## Count the number of files in a directory

### Windows

1. Run the following command to get the count for each directory:

````text
Get-ChildItem -Path "D:\OneDrive\Master Thesis\datasets\raw\synthetic-data" -Directory | ForEach-Object {
    $images = Get-ChildItem -Path "$($_.FullName)\images" -File
    Write-Output "$($_.Name): $($images.Count) file(s)"
}
````

2. Run the following command to get the total count:

````text
$rootDir = "D:\OneDrive\Master Thesis\datasets\raw\synthetic-data"
$totalCount = 0

Get-ChildItem -Path $rootDir -Directory | ForEach-Object {
    $imagesPath = Join-Path $_.FullName "images"
    if (Test-Path $imagesPath) {
        $totalCount += (Get-ChildItem -Path $imagesPath -File -ErrorAction SilentlyContinue).Count
    }
}
Write-Output "Total files in all 'images' folders: $totalCount"
````

### Linux

1. Run the following command to get the total count:

````text
find /mnt/shared/datasets/raw/synthetic-data/*/images -maxdepth 1 -type f 2>/dev/null | wc -l
````