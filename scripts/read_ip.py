import json

with open('./scripts/.ec2-info.json', 'r') as f:
    data = json.load(f)

instance = data['Reservations'][0]['Instances'][0]
ip = instance['NetworkInterfaces'][0]['Association']['PublicIp']
with open('./scripts/.ec2-ip.sh', 'w') as f:
    f.write(f'EC2_IP="{ip}"')
