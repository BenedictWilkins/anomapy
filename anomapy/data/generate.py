import numpy as np

import pyworld.toolkit.tools.gymutils as gu
from pyworld.toolkit.tools import fileutils as fu
import pyworld.toolkit.tools.visutils as vu

from anomapy import load

PATH = '~/Documents/repos/datasets/gym/{0}/'

def generate(env, policy=None, path=PATH, train_episodes=100, test_episodes=10):
    env_id = env.unwrapped.spec.id

    train_path = path.format(env_id)
    test_path = path.format(env_id) + "test/"
    
    print("GENERATING DATA FOR ENVIRONMENT: ", env_id)
    print("   PATH:", train_path)
    print("   EPISODES: TRAIN:", train_episodes, "TEST:", test_episodes)

    env = gu.wrappers.EpisodeRecordWrapper(env, train_path)
    for i in range(train_episodes):
        gu.episode(env, policy)
    env.path = test_path
    for i in range(test_episodes):
        gu.episode(env, policy)
        
        
def raw_to_clean(env):
    t = len(load.files_raw(env))
    i = 1
    for file, episode in load.load_raw(env):
        print("ENV: {0:<16} processing file: {1:<2}/{2:<2}".format(env, i, t))
        
        episode['state'] = episode['state'].astype(np.float32) / 255.
        episode['state'] = vu.transform.CHW(episode['state']) 
        clean_file = file.replace('raw', 'clean')
        #vu.play(episode['state'])
        
        fu.save(clean_file, episode)
        i += 1
