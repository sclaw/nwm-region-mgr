# !/bin/bash

# To load and run the docker image:
# 1. download from s3://ngwpc-dev/jeff.wade/docker/mswm.tar.gz
# 2. unpack: gunzip mswm.tar.gz
# 3. load the image: docker load -i mswm.tar
# 4. run this script: ./run_msw_docker.sh

sudo docker run -it \
    --env TERM=xterm-256color \
    --entrypoint /bin/bash \
    -v /home/yuqiong.liu/:/home/yuqiong.liu:Z \
    mswm \
    -c "echo \"alias ls='ls --color=auto'\" >> ~/.bashrc && \
        cd /home/yuqiong.liu/repos/nwm-region-mgr && \
        /bin/bash -l"
