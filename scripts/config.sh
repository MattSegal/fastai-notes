# Config for AWS Ubuntu 18.04 AMI
# Prepare to instal nvidia + cuda
set -e
apt update
apt install -y \
    build-essential \
    software-properties-common \
    ubuntu-drivers-common

ubuntu-drivers autoinstall

add-apt-repository -y ppa:ubuntu-toolchain-r/test

apt update
apt install -y \
    gcc-6 \
    g++-6

update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-6

# Check GCC version
gcc -v

# Install nvidia + cuda
apt install nvidia-cuda-toolkit

# Check graphics card
nvidia-smi

# Check CUDA version
nvcc --version

apt install -y \
    git \
    python3-pip \
    virtualenv

virtualenv -p python3 env

. ~/env/bin/activate

pip3 install \
    jupyter \
    torch \
    torchvision \
    torchtext \
    dill \
    pillow \
    requests \
    numpy \
    bcolz \
    scipy \
    opencv-python \
    spacy \
    pandas \
    seaborn \
    graphviz \
    sklearn_pandas \
    sklearn  \
    isoweek \
    pandas_summary \
    tqdm \
    matplotlib

python3 -m spacy download en

git clone https://github.com/fastai/courses.git
