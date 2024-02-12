'''
 # @ Author: Y. Xiao
 # @ Create Time: 2024-02-11 19:48:46
 # @ Modified by: Y. Xiao
 # @ Modified time: 2024-02-11 19:48:58
 # @ Description: Utils for the project.
 '''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Tuple
import math
from enum import Enum


def clone(
    module: nn.Module, 
    copyNum: int
) -> nn.ModuleList:
    """Produce N identical layer from the initial layer."""
    return nn.ModuleList([deepcopy(module) for _ in range(copyNum)])