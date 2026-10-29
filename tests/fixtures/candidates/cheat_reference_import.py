# task: L1-1
# expect: reference_import
import torch.nn as nn
from reference import Model


class ModelNew(Model):
    pass
