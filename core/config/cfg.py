import yaml
import os
from easydict import EasyDict as edict

# cfg_name = '160X96_crop++_shuffle_lite-hrnet18_sa-simdr4_stgcn++.yaml'
#
# cfg_name = '160X96_crop++_lite-hrnet18_sa-simdr4_stgcn++.yaml'
# #
# cfg_name = '128X128_crop++_shuffle_lite-hrnet18++_sa-simdr4_stgcn++.yaml'
#
# cfg_name = 'abla/baseline.yaml'
# #
# cfg_name = 'abla/setting1.yaml'
# # # # # # # # # # # # #
# cfg_name = 'abla/setting2.yaml'
# # # # # # # # # # #
# cfg_name = 'abla/setting3.yaml'
# # # # #
# cfg_name = 'abla/setting4.yaml'
# # cfg_name = 'accimproving/accimproving_320X192cv.yaml'
# cfg_name = 'abla/setting6.yaml'

# cfg_name = 'loss/loss_setting2.yaml'

# cfg_name = 'loss/loss_setting3.yaml'

# cfg_name = 'baseline/baseline1.yaml'

# cfg_name = 'loss/loss_setting1.yaml'
#
# cfg_name = 'loss/loss_setting2.yaml'
# #
# cfg_name = 'test/test3.yaml'
#
# cfg_name = 'baseline/baseline1_cs.yaml'
#
# cfg_name = 'baseline/baseline3.yaml'

# cfg_name = 'switch/switch2_big.yaml'
#
# cfg_name = 'lr/lr_sgd.yaml'
#
# cfg_name = 'lr/adam_cycle1.yaml'
#
# cfg_name = 'lr/sgd_cycle2.yaml'
# #
# cfg_name = 'lr/sgd_cycle3.yaml'

# cfg_name = 'lr/sgd_cycle5.yaml'

# cfg_name = 'lr/sgd_cycle5_1.yaml'

# cfg_name = 'lr/sgd_cycle5_2.yaml'

# cfg_name = 'lr/sgd_cycle6.yaml'

# cfg_name = 'lr/sgd_cycle6_1.yaml'

# cfg_name = 'lr/sgd_cycle7.yaml'
# #
# cfg_name = 'lr/sgd_cycle7_1.yaml'

# cfg_name = 'lr/sgd_cycle7_2.yaml'

cfg_name = 'lr/sgd_cycle8.yaml'

#
with open(os.path.join(os.path.dirname(__file__), cfg_name), 'r') as f:
    cfg = yaml.safe_load(f)
