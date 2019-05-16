FROM anibali/pytorch:cuda-10.0
USER root
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak \
    && echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse > /etc/apt/sources.list \
    && echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse >> /etc/apt/sources.list \
    && echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse >> /etc/apt/sources.list \
    && echo deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse >> /etc/apt/sources.list \
    && apt update
RUN buildDeps='gcc libsm6 libxext6 vim tmux screen' \
    && apt-get install -y $buildDeps --fix-missing
RUN mkdir /home/nfs
RUN groupadd -g 1004 SR_group
USER user
RUN pythonPkg='numpy opencv-python lmdb tensorboardX' \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install $pythonPkg
