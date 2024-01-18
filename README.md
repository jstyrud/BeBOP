# BeBOP
This is the project repository for the paper 

**BeBOP - Combining Reactive Planning and Bayesian Optimization to Solve Robotic Manipulation Tasks**

submitted to IEEE International Conference on Robotics and Automation (ICRA), 2024

A preview version can be found at:
https://arxiv.org/abs/2310.00971

## Installation

# Docker
This project is built as a docker container. To install docker, [follow the instructions](https://docs.docker.com/engine/install/ubuntu/). The image can be pulled with:
```shell
docker pull matthiasmayr/bebop:main
```

It can be run with:
```shell
docker run -it --rm --tag bebop matthiasmayr/bebop:main
```
If you want to spawn the simulation 


# Bare Metal
Create a project folder and clone this repo
```
apt install git wget
mkdir bebop && cd bebop
git clone git@github.com:jstyrud/BeBOP.git
```

Execute the installation script:
```
bebop/scripts/install.sh
```
