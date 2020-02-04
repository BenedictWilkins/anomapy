import load
import video_anomaly

import pyworld.toolkit.tools.gymutils as gu
from pyworld.toolkit.tools import fileutils as fu
import pyworld.toolkit.tools.visutils as vu

import numpy as np

import gym

if __name__ == "__main__":
    for env in load.ENVIRONMENTS:
        env_noframeskip = env + "NoFrameskip-v4"
        print("{0}: {1}".format(env_noframeskip, gym.make(env_noframeskip).action_space))
        f,e = next(load.load_clean(env, limit=1000))
        print(np.unique(e['action']))
