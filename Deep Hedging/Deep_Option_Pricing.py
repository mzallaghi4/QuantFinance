# ============================================================
# DEEP LEARNING FOR EUROPEAN CALL OPTION PRICING
# Model: Fully Connected Neural Network (MLP)
#
# Inputs:  Spot (S), Strike (K), Maturity (T)
# Output:  European Call Option Price (C)
#
# Target transformation: C / S
# Final price: S * predicted(C / S)
# ============================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
