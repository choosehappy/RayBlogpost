# Ray: An Open-Source API for Easy, Scalable Distributed Computing in Python

This repo contains the Ray Core and Ray Serve code described in [this blog post](http://www.andrewjanowczyk.com/).


## Run ray_core script

Create an image:

```
    docker build -t ray/core:1.0.0 -f DockerfileCore .
```

Run a container built from the image

```
    docker run --name ray-core -it ray/core:1.0.0
```



## Run ray_serve script

Create an image:


```
    docker build -t ray/server:1.0.0 -f DockerfileServer .
```

Run a container built from the image


```
    docker run --name ray-server -it ray/server:1.0.0
```
