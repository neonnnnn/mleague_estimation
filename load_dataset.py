import json
import glob
import os
import numpy as np
from scipy.stats import rankdata


point_bias = [45, 5, -15, -35]


def _add_ranking_point(y_original, sign):
    y = np.array(y_original)
    scores_rank = rankdata(-y, axis=1)
    for rank in np.unique(scores_rank):
        bias = (point_bias[int(rank)-1] + point_bias[int(rank+0.5)-1]) / 2
        y[scores_rank == rank] += sign * bias
    if not np.all(scores_rank == rankdata(-y, axis=1)):
        raise ValueError("Wrong rank point operation.")
    return y


def remove_ranking_point(y):
    return _add_ranking_point(y, -1)


def add_ranking_point(y_original):
    return _add_ranking_point(y_original, 1)


def load_mleague_dataset(file_dir="./data/"):
    files = glob.glob(os.path.join(file_dir, "*.json"))
    results = []
    for file in files:
        with open(file) as f:
            results.append(json.load(f))

    players_all = []
    scores_all = []
    for result in results:
        players = []
        scores = []
        for key, val in result.items():
            players.append(key)
            scores.append(val)
        players_all.append(players)
        scores_all.append(scores)

    y = np.array(scores_all)
    return np.array(players_all), y
