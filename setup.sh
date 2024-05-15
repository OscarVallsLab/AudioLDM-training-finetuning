export DEBIAN_FRONTEND=noninteractive export TZ=Europe/Madrid
export PYTHONPATH=$PYTHONPATH:/workspace/code
apt update
apt install -y python3 python3-pip python3-tk
pip install --upgrade pip
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.1.1
pip install -r requirements.txt