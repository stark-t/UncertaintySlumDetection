import config as config
import pandas as pd
import numpy as np


def preprocess_input(x, **kwargs):
    image_stats = pd.read_csv(config.image_stats_path)

    mean = image_stats.groupby(["channel"])["mean"].mean().tolist()
    std = image_stats.groupby(["channel"])["std"].mean().tolist()

    x = x / 255.0
    mean = np.divide(mean, 255.0)
    std = np.divide(std, 255.0)

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    # x = x / 255.0

    return x
