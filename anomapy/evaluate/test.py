from . import evaluate
from .. import video_anomaly
from .. import load

import pyworld.toolkit.tools.visutils as vu

def test_split():
    for env in load.ENVIRONMENTS:
        for anom in video_anomaly.ANOMALIES:
            evaluate.split_subsequence(env, anom)

test_split()

import numpy as np


