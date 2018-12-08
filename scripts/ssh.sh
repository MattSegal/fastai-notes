#/bin/bash
set -e
. ./scripts/.ec2-ip.sh
ssh  -L localhost:8888:localhost:8888 root@$EC2_IP
