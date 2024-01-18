#!/bin/bash

# Author: Matthias Mayr
# Date: October 2023
# Purpose: Install dependencies for running BeBOP
# Installs MAPLE, robotsuite, mujoco, a special version of mujoco-py & a pytrees fork

# Configuration
MUJOCO_V=210 # Mujoco Version

sudo apt install -y wget python3-pip
# Install maple
git clone https://github.com/UT-Austin-RPL/maple &

# Install robosuite with MuJoCo
git clone -b maple https://github.com/ARISE-Initiative/robosuite &
mkdir ~/.mujoco || true
if [ "$MUJOCO_V" == "200" ];
then
  sudo apt-get update
  sudo apt install -y unzip
  wget https://www.roboti.us/download/mujoco200_linux.zip
  unzip mujoco200_linux.zip
  mv mujoco200_linux ~/.mujoco/mujoco$MUJOCO_V || true
  # Activation key
  wget https://www.roboti.us/file/mjkey.txt && mv mjkey.txt ~/.mujoco/
elif [ "$MUJOCO_V" == "210" ];
then
  wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar -xf mujoco210-linux-x86_64.tar.gz
  mv mujoco210 ~/.mujoco/mujoco$MUJOCO_V
fi
# Dependencies
sudo apt install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common net-tools xpra xserver-xorg-dev libglfw3-dev patchelf
string="export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco$MUJOCO_V/bin"
temp=$(cat ~/.bashrc | grep "$string")
if [ -z "$temp" ]; then
    echo "# MuJoCo path" >> ~/.bashrc
	echo $string >> ~/.bashrc
	echo "Added string in .bashrc: $string"
else
	echo "No export string added to bashrc (it is already there)"
fi
# This is needed for the docker build:
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco$MUJOCO_V/bin

# Install and set up anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh -b -p $HOME/anaconda3
string="export PATH=$HOME/anaconda3/bin:\$PATH"
temp=$(cat ~/.bashrc | grep "$string")
if [ -z "$temp" ]; then
  echo "# Anaconda" >> ~/.bashrc
	echo $string >> ~/.bashrc
	echo "Added string in .bashrc: $string"
else
	echo "No export string added to bashrc (it is already there)"
fi
source ~/.bashrc
# This is needed in the docker build:
export PATH=/root/anaconda3/bin:$PATH
conda init bash

# Fix MuJoCo setup
# Container fix to build mujoco-py
pip install Cython==3.0.0a10
git clone https://github.com/nimrod-gileadi/mujoco-py.git
cd mujoco-py
pip install --user .
cd ..

# Wait for dead-slow git clones to finish before continuing
FAIL=0
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
if [ "$FAIL" != "0" ];
then
  echo "FAIL! Background jobs failed. Check logs for $FAIL failures."
fi

cd maple
conda env create --name maple --file=maple.yml
sed -i 's/mujoco-py==2.0.2.9/mujoco-py==2.1.2.14/g' maple.yml
sed -i 's/numpy==1.19.5/numpy==1.22.4/g' maple.yml
conda env update --name maple --file=maple.yml
conda activate maple
pip install -e . &
cd ..

cd robosuite
# Fix mujoco-py to 2.1.2.14
sed -i 's/mujoco-py==2.0.2.9/mujoco-py==2.1.2.14/g' requirements.txt
sed -i 's/mujoco-py==2.0.2.9/mujoco-py==2.1.2.14/g' setup.py
pip install -e . &
pip3 install --user -r requirements-extra.txt
cd ..

git clone git@github.com:jstyrud/py_trees.git
cd py_trees
pip install .
cd ..

git clone --branch hypermapper-v3 https://github.com/luinardi/hypermapper/tree/hypermapper-v3 hypermapper
cd hypermapper
pip install .
cd ..

cd bebop
pip install -r requirements.txt
cd ..

# Adds custom packages to PYTHONPATH
string="export PYTHONPATH=$PYTHONPATH:`pwd`/bebop:`pwd`/py_trees:`pwd`/hypermapper"
echo $string >> ~/.bashrc

# Wait for dead-slow finish
FAIL=0
for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done
if [ "$FAIL" != "0" ];
then
  echo "FAIL! Background jobs failed. Check logs for $FAIL failures."
fi
