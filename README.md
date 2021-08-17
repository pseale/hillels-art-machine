
## 👷👩‍🏭👷👷👩‍🏭👨‍🏭👷👩‍🏭👨‍🏭👷👨‍🏭👨‍🏭👷👷👨‍🏭👨‍🏭👨‍🏭👩‍🏭👷👷👷👷👨‍🏭👷👷👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👷👩‍🏭👷👨‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👷👨‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭
## 👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👷👷👷👩‍🏭👨‍🏭👷👷👷👩‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👷👨‍🏭👷👨‍🏭👷👨‍🏭👩‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👷👷👨‍🏭👩‍🏭👨‍🏭👨‍🏭👷👷👨‍🏭👩‍🏭👷👷👨‍🏭👷👨‍🏭👨‍🏭👷👨‍🏭👷
## 👨‍🏭👷👷👷👷👨‍🏭👷👩‍🏭👨‍🏭👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👷👩‍🏭👩‍🏭👷👨‍🏭👷👷👷👩‍🏭👷👩‍🏭👷👷👨‍🏭👷👷👨‍🏭👷👨‍🏭👷👷👩‍🏭👩‍🏭👷👩‍🏭👨‍🏭👩‍🏭
## WORK IN PROGRESS
## 👩‍🏭👷👷👨‍🏭👷👩‍🏭👩‍🏭👷👩‍🏭👷👩‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👩‍🏭👷👩‍🏭👩‍🏭👩‍🏭👷👷👩‍🏭👩‍🏭👷👷👩‍🏭👷👨‍🏭👩‍🏭👷👨‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👷👩‍🏭👷👷👩‍🏭👨‍🏭👷👷👷👩‍🏭👷👨‍🏭👨‍🏭
## 👨‍🏭👷👷👩‍🏭👨‍🏭👨‍🏭👨‍🏭👷👨‍🏭👷👨‍🏭👨‍🏭👨‍🏭👷👩‍🏭👨‍🏭👷👨‍🏭👷👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭👨‍🏭👷👷👷👨‍🏭👨‍🏭👷👩‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭👷👷👨‍🏭👷👩‍🏭👩‍🏭👷
## 👷👨‍🏭👩‍🏭👷👷👩‍🏭👩‍🏭👨‍🏭👨‍🏭👩‍🏭👩‍🏭👷👷👩‍🏭👷👷👨‍🏭👨‍🏭👨‍🏭👩‍🏭👷👷👩‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👷👨‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👩‍🏭👩‍🏭👷👷👨‍🏭👨‍🏭

This is an attempt to make a CLI version of @hillel on twitter's Art Machine, as found here https://colab.research.google.com/drive/1n_xrgKDlGQcCF6O-eL3NOd_x4NSqAUjK - which itself is based on another notebook by Katherine Crowson.

Anyway, what I'm saying is I didn't build this. I'm attempting to package the thing other people built into a CLI tool.

### Install instructions that I have not tested, on Ubuntu 20.04 via WSL2 on Windows

First get the Insiders update that enables CUDA. https://forums.developer.nvidia.com/t/nvidia-smi-through-wsl2/180310/6

Then:

``` bash

git clone git@github.com:pseale/hillels-art-machine.git
cd hillels-art-machine
sudo apt install nvidia-340 # check if this is correct - it was as of 2021-08-16
sudo apt install python3-pip
pip install pipenv
pipenv install
curl -C - -o vqgan_imagenet_f16_1024.yaml -L 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml'
curl -C - -o vqgan_imagenet_f16_1024.ckpt -L 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt'

# finally run the tool with default parameters
# TODO: everything 👍
pipenv run ./ham --text-prompt "tree bark flavored ice cream" # choose better words

# or, provide an optional image prompt
pipenv run ./ham --text-prompt "swirling clouds and a glorious sunrise" --image-prompt 'https://pbs.twimg.com/media/E8DTA6MXEAIHVE_?format=jpg&name=4096x4096'
```