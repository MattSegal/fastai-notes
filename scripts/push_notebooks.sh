#/bin/bash
set -e
. ./scripts/.ec2-ip.sh

ssh root@$EC2_IP /bin/bash << EOF
    mkdir -p ~/notebooks
EOF

scp ./notebooks/* root@${EC2_IP}:/root/notebooks/
scp ./scripts/run_jupyter.sh root@${EC2_IP}:/root/
