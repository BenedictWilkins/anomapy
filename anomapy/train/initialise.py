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

from pyworld.toolkit.tools.wbutils import WB as wb_init

def WB(_model, **kwargs):
    return wb_init('anomapy', _model, id = "{0}-{1}-{2}".format(kwargs['model'], kwargs['env'], fu.file_datetime()), tags=[kwargs['model']], config=kwargs)

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
    parser.add_argument("-colour", type=bool, default=True, help="whether to use full colour states (True) or transform states to grayscale (False)")    

    args = parser.parse_args()

    args.__dict__['model'] = model
    args.__dict__.update(load.HYPER_PARAMETERS[args.env]) #update any hyper params that have not yet been given

    print("--------------------------")
    print(args.env, flush=True)
    print("--------------------------")

    print("-- loading data")

    data_size = 0
    episodes = []
    for file, episode in load.load_clean(args.env, limit=args.dataset_size):
        print("---- loading episode {0}".format(file))
        data_size += episode['state'].shape[0]
        episodes.append(load.transform(episode, args))
    print("-- done, total frames: {0}".format(data_size))

    args.__dict__['state_shape'] = tuple(episodes[0]['state'].shape[1:])
    if not args.env in load.ACTION_SHAPES:
        raise ValueError("could not find action shape for environment: {0}".format(args.env))
    args.__dict__['action_shape'] = (load.ACTION_SHAPES[args.env],)

    args.latent_shape = (args.latent_shape, )

    if args.colour:
        assert args.state_shape[0] == 3
    else:
        assert args.state_Shape[0] == 1

    return episodes, args

def states(episodes, shuffle=True):
    print("-- preparing episodes...")
    print("---- {0:<3} train episodes".format(len(episodes)))
    episodes = [e['state'] for e in episodes]
    if shuffle:
        for episode in episodes:
            np.random.shuffle(episode)
    episodes = [torch.from_numpy(e) for e in episodes]
    print("-- done.")
    return episodes

def states_actions(episodes):
    print("-- preparing episodes...")
    print("---- {0:<3} train episodes".format(len(episodes)))

    states = [torch.from_numpy(e['state']) for e in episodes]
    actions = [e['action'] for e in episodes]

    episodes =  list(zip(states, actions))

    print("-- done.")
    return episodes

def load_mnist():
    x_train, _, x_test, _ = du.mnist()
    return torch.from_numpy(x_train), torch.from_numpy(x_test), x_train.shape[1:]