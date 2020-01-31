import numpy as np
import torch

import argparse
from pprint import pprint

from .. import load

import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.debugutils as debug

def initialise(model):
    if not model in load.MODEL:
        raise ValueError("Invalid model {0}, valid models include: {1}".format(model, load.MODEL))

    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, required=True)    
    parser.add_argument("-latent_shape", type=int, default=256)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-dataset_size", type=int, default=None, help="how much data to use frames. Default None means use all avaliable data.") #use all data
    parser.add_argument("-device", type=str, default = tu.device(), help="which device to use (default GPU if avaliable).")    
    parser.add_argument("-colour", type=bool, default=False, help="whether to use full colour states (True) or transform states to grayscale (False)")    
    
    args = parser.parse_args()

    args.__dict__['model'] = model
    args.__dict__.update(load.HYPER_PARAMETERS[args.env]) #update any hyper params that have not yet been given

    print("--------------------------")
    print(args.env, flush=True)
    print("--------------------------")

    print("-- loading data")

    episodes = []
    for file, episode in load.load_clean(args.env, limit=args.dataset_size):
        print("---- loading episode {0}".format(file))
        episodes.append(load.transform(episode, args))
    print("-- done.")

    args.__dict__['state_shape'] = tuple(episodes[0]['state'].shape[1:])
    args.__dict__['action_shape'] = (1,) #should always be 1?

    return episodes, args

def states(episodes, test_episodes=1):
    assert test_episodes < len(episodes)
    print("-- preparing episodes...")
    print("---- {0:<3} train episodes".format(len(episodes) - test_episodes))
    print("---- {0:<3}  test episodes".format(test_episodes))
    #only states are required
    episodes = [e['state'] for e in episodes]
    
    for episode in episodes[-test_episodes]:
        np.random.shuffle(episode)
    episodes = [torch.from_numpy(e) for e in episodes]

    episode_test = episodes[-test_episodes] 
    episodes = episodes[:-test_episodes]
    print("-- done.")

    return episodes, episode_test