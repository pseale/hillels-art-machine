
## 👷👩‍🏭👷👷👩‍🏭👨‍🏭👷👩‍🏭👨‍🏭👷👨‍🏭👨‍🏭👷👷👨‍🏭👨‍🏭👨‍🏭👩‍🏭👷👷👷👷👨‍🏭👷👷👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👷👩‍🏭👷👨‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👷👨‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭
## 👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👷👷👷👩‍🏭👨‍🏭👷👷👷👩‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👷👨‍🏭👷👨‍🏭👷👨‍🏭👩‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👷👷👨‍🏭👩‍🏭👨‍🏭👨‍🏭👷👷👨‍🏭👩‍🏭👷👷👨‍🏭👷👨‍🏭👨‍🏭👷👨‍🏭👷
## 👨‍🏭👷👷👷👷👨‍🏭👷👩‍🏭👨‍🏭👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👷👩‍🏭👩‍🏭👷👨‍🏭👷👷👷👩‍🏭👷👩‍🏭👷👷👨‍🏭👷👷👨‍🏭👷👨‍🏭👷👷👩‍🏭👩‍🏭👷👩‍🏭👨‍🏭👩‍🏭
## WORK IN PROGRESS
## 👩‍🏭👷👷👨‍🏭👷👩‍🏭👩‍🏭👷👩‍🏭👷👩‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👩‍🏭👷👩‍🏭👩‍🏭👩‍🏭👷👷👩‍🏭👩‍🏭👷👷👩‍🏭👷👨‍🏭👩‍🏭👷👨‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👷👩‍🏭👷👷👩‍🏭👨‍🏭👷👷👷👩‍🏭👷👨‍🏭👨‍🏭
## 👨‍🏭👷👷👩‍🏭👨‍🏭👨‍🏭👨‍🏭👷👨‍🏭👷👨‍🏭👨‍🏭👨‍🏭👷👩‍🏭👨‍🏭👷👨‍🏭👷👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👨‍🏭👨‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭👨‍🏭👷👷👷👨‍🏭👨‍🏭👷👩‍🏭👩‍🏭👩‍🏭👩‍🏭👨‍🏭👷👷👨‍🏭👷👩‍🏭👩‍🏭👷
## 👷👨‍🏭👩‍🏭👷👷👩‍🏭👩‍🏭👨‍🏭👨‍🏭👩‍🏭👩‍🏭👷👷👩‍🏭👷👷👨‍🏭👨‍🏭👨‍🏭👩‍🏭👷👷👩‍🏭👨‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👩‍🏭👷👨‍🏭👨‍🏭👩‍🏭👨‍🏭👷👨‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👩‍🏭👨‍🏭👩‍🏭👩‍🏭👩‍🏭👷👷👨‍🏭👨‍🏭

This is an attempt to make a CLI version of @hillelogram on twitter's Art Machine, as found here https://colab.research.google.com/drive/1n_xrgKDlGQcCF6O-eL3NOd_x4NSqAUjK - which itself is based on another notebook by Katherine Crowson (@rivershavewings).

Anyway, what I'm saying is I didn't build this. I'm attempting to package the thing other people built into a CLI tool.

### CPU vs GPU

This will work against either a CPU or GPU. Running the same job against a CPU took 200 minutes versus 4 running on the GPU.

My GPU with 8GB of RAM could only support a 400x400 image.

### Install Instructions for Ubuntu 20.04

First get the Insiders update that enables CUDA. https://forums.developer.nvidia.com/t/nvidia-smi-through-wsl2/180310/6

Then:

``` bash

git clone git@github.com:pseale/hillels-art-machine.git
cd hillels-art-machine

nvidia-smi # if this doesn't run, install nvidia drivers

sudo apt install python3-pip
pip install pipenv
pipenv install
curl -C - -o vqgan_imagenet_f16_1024.yaml -L 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml'
curl -C - -o vqgan_imagenet_f16_1024.ckpt -L 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt'

# finally run the tool with default parameters
./ham --text-prompt "an apple and some grapes and a banana in a wood bowl"

# or, provide an optional image prompt
./ham --text-prompt "swirling clouds and a glorious sunrise" --image-prompt 'https://pbs.twimg.com/media/E8DTA6MXEAIHVE_?format=jpg&name=4096x4096'
```
