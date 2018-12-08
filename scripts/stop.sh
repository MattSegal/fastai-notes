#!/bin/bash
. "$(dirname $0)/settings.sh"
echo "Stopping EC2"
aws ec2 stop-instances --instance-ids $INSTANCE_ID
echo "EC2 Status"
aws ec2 describe-instance-status --instance-ids $INSTANCE_ID
