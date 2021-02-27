import numpy as np
import pandas as pd
import requests
import csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

from scipy import stats
from scipy.interpolate import interp1d
from scipy import optimize

import seaborn as sns

from tqdm.notebook import tqdm

import torch.nn as nn

print("\nTest passed!\n")
