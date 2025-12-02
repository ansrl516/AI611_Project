FROM continuumio/miniconda3

WORKDIR /workspace

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-gfx-dev \
    swig \
    wget \
    git

# # Boost.Python 3.9 빌드 및 설치
# RUN wget https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz && \
#     tar -xzf boost_1_83_0.tar.gz && \
#     cd boost_1_83_0 && \
#     ./bootstrap.sh --with-libraries=python && \
#     ./b2 python=3.9 && \
#     cp stage/lib/libboost_python39.* /usr/lib/x86_64-linux-gnu/ && \
#     cd .. && rm -rf boost_1_83_0 boost_1_83_0.tar.gz

# 환경 파일 및 코드 복사
COPY environment.yml ./
COPY . /workspace

# Conda 환경 생성
RUN conda env create -f environment.yml

# Boost.Python 빌드 (conda 환경의 python 사용. python3.9 또한 conda 환경 내 존재)
RUN /bin/bash -c "source activate zsceval && \
    PYTHON_BIN_PATH=$(which python) && \
    cd /tmp && \
    wget -O boost_1_83_0.tar.gz https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.gz && \
    tar -xzf boost_1_83_0.tar.gz && \
    cd boost_1_83_0 && \
    ./bootstrap.sh --with-libraries=python --with-python=$PYTHON_BIN_PATH && \
    ./b2 python=3.9 && \
    cp stage/lib/libboost_python39.* /usr/lib/x86_64-linux-gnu/ && \
    cd .. && rm -rf boost_1_83_0 boost_1_83_0.tar.gz"

# gfootball 설치
# RUN /bin/bash -c "source activate zsceval && pip install psutil && pip install gfootball[extras]"
RUN /bin/bash -c "source activate zsceval && pip install psutil"
# 기본 쉘을 conda 환경으로 설정
SHELL ["conda", "run", "-n", "zsceval", "/bin/bash", "-c"]

# 사용자 및 그룹 생성
RUN groupadd -g 1015 moongichoi && \
    useradd -u 1015 -g moongichoi -m moongichoi

CMD ["/bin/bash"]