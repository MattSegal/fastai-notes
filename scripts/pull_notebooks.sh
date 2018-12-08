#/bin/bash
set -e
. ./scripts/.ec2-ip.sh
scp \
    root@$EC2_IP:/root/notebooks/*.ipynb \
    notebooks
