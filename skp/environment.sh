# Miniconda
xport ENV_NAME="skp2"
conda create -n $ENV_NAME -y python=3.11.5 pip
conda activate $ENV_NAME
conda install -y pytorch=2.1.1 torchvision=0.16.1 torchaudio=2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y lightning=2.1.2 timm=0.9.10 transformers=4.35.2 scikit-learn=1.3.2 monai=1.3.0 pandas=2.1.3 pydicom=2.4.3 gdcm=2.8.9 kaggle=1.5.16 -c conda-forge
pip install neptune==1.8.3 pytorchvideo==0.1.5 albumentations==1.3.1
