FROM nvidia/cuda:9.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install PyTorch with cuda-9.2 support
RUN conda install -y -c pytorch \
    cuda92=1.0 \
    magma-cuda92=2.3.0 \
    "pytorch=0.4.1=py36_cuda9.2.148_cudnn7.1.4_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install JupyterLab
RUN conda install -y jupyterlab=0.33.4 \
 && conda clean -ya

# Install other dependencies from pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user . /app
RUN pip install -e .

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
