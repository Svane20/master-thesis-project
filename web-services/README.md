## Docker Compose

1. Go to the root of the project
2. Run the following command

````bash
docker compose up --build -d
````

## Add tracking to Grafana

1. Go to Grafana

````text
http://localhost:3000
````

2. Login

````text
username: admin
password: admin
````

3. Click on Connections -> Data sources
4. Click on Add data source
5. Select Prometheus
6. Fill in the URL of the Prometheus server

````text
http://prometheus:9090
````

7. Click on Save & Test