# docker build -t moongi_zsceval .

docker run --gpus all --shm-size=3g --name moongi_zsceval_container \
    -v $(pwd):/workspace \
    -it moongi_zsceval bash