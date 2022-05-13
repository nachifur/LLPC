import os
import yaml
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help="gpu")
ARGS = parser.parse_args()
gpu = ARGS.gpu



fr = open("config.yml.example", 'r')
config = yaml.load(fr, Loader=yaml.FullLoader)
config["DEBUG"] = 0
config["GPU"] = [gpu]
with open("config.yml.example", 'w') as f_obj:
    yaml.dump(config, f_obj)
print("in deploy mode")

# clear
# model_name = config["MODEL_NAME"] + '.pth'
# checkpoints_path = str(Path('./checkpoints') / \
#     config["SUBJECT_WORD"]/model_name)
# flag = os.system("rm "+checkpoints_path)
# if flag == 0:
#     print("clear "+checkpoints_path+" success")
flag = os.system("rm -rf src/__pycache__")
if flag == 0:
    print("clear src/__pycache__ success")

