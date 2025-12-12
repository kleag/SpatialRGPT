#!/usr/bin/env bash


# This is required to enable PEP 660 support
uv pip install --upgrade pip

# Install FlashAttention2
uv pip install ~/flash_attn-2.8.3+cu130torch2.9-cp310-cp310-linux_x86_64.whl


# Install VILA
uv pip install -e .
uv pip install -e ".[train]"
uv pip install -e ".[eval]"

# Install HF's Transformers
pip install git+https://github.com/huggingface/transformers@v4.37.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
