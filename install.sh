conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip3 install "torch<2.8.0" "torchvision<0.23.0" --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[smolvla, pusht, libero, uncertainty]"
