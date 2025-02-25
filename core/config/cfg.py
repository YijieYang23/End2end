import yaml
import os
from easydict import EasyDict as edict

cfg_name = 'baseline/baseline_cv.yaml'

with open(os.path.join(os.path.dirname(__file__), cfg_name), 'r') as f:
    cfg = yaml.safe_load(f)
