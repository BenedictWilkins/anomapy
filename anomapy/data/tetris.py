from nes_py.wrappers import JoypadSpace
import gym_tetris
import gym
from gym_tetris.actions import SIMPLE_MOVEMENT

import numpy as np

from pyworld.toolkit.tools import gymutils as gu
from pyworld.toolkit.tools import visutils as vu
from pyworld.toolkit.tools import fileutils as fu

CROP_BOX = (95, 47, 176,209)
SAVE_PATH = "~/Documents/repos/datasets/tetris/episode.hdf5"
EPISODES = 38

env = gym.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT) #only 6 actions please!
env = gu.wrappers.ObservationWrapper(env, mode=gu.transformation.mode.crop, box=CROP_BOX)

env = gu.wrappers.EpisodeRecordWrapper(env, SAVE_PATH)
policy = gu.policy.uniform_random_policy(env.action_space)

#TEST RNG
#state1 = fu.load("~/Documents/repos/datasets/tetris/episode.hdf5")['state'][...]
#state2 = fu.load("~/Documents/repos/datasets/tetris/episode(1).hdf5")['state'][...]
#state = np.concatenate((state1[:9000:20], state2[:9000:20]), 2)
#vu.play(state, wait=3)

for _ in range(EPISODES):
    gu.episode(env, policy=policy)

#for i in range(1, EPISODES):
#    states = fu.load("~/Documents/repos/datasets/tetris/episode({0}).hdf5".format(i))['state'][...]
#    vu.play(states)




#print(env.observation_space)

#print(env.observation_space.shape)
#state = next(gu.video(env))
#fu.save("./teris.png", state)

#vu.play(gu.video(env))