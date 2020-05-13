from pyworld.algorithms.optimise.TripletOptimiser import SSTripletOptimiser, mode as optim_mode
from pyworld.toolkit.nn.CNet import CNet2

import pyworld.toolkit.tools.wbutils as wbu
import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.debugutils as debug

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import copy


from types import SimpleNamespace
from pprint import pprint

from .. import load
from . import initialise
from ..evaluate import evaluate

from . import utils

import datasets

DEFAULT_CONFIG = dict(model='sssn', latent_shape=2,
                      epochs=10, batch_size=64, device=tu.device(False),
                      optim_mode=optim_mode.all, optim_margin=0.2, 
                      optim_k=16, optim_lr=0.0005)

def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)

def new(**config):

    _config = default_config()
    _config.update(config)
    config = _config

    model = CNet2(config['state']['shape'], config['latent_shape']).to(config['device'])
    optim = SSTripletOptimiser(model, mode=config['optim_mode'], margin=config['optim_margin'], k=config['optim_k'], lr=config['optim_lr'])

    config["model_class"] = model.__class__.__name__
    config["optim_class"] = optim.__class__.__name__

    return optim

def model(**config):
    return CNet2(config['state']['shape'], config['latent_shape']).to(config['device'])

def optimiser(**config):
    return SSTripletOptimiser(model(**config), mode=config['optim_mode'], margin=config['optim_margin'], k=config['optim_k'], lr=config['optim_lr'])

def distance(model, episode, batch_size=256):
    model.eval()
    z = tu.collect(model, episode, batch_size=batch_size) #how much memory has the gpu got?!
    model.train()
    #L22 distance by default? TODO give some other options for this
    d = ((z[:-1] - z[1:]) ** 2).sum(1)
    return z, d

def encode(model, episode, batch_size=256):
    model.eval()
    z = tu.collect(model, episode, batch_size=batch_size) #how much memory has the gpu got?!
    model.train()
    return z



def epoch(optimiser, episode, batch_size=64):
    for batch in du.batch_iterator(episode[:-1], episode[1:], batch_size=batch_size, shuffle=True):
        batch_state, batch_next_state = batch
        optimiser(batch_state, batch_next_state)
        yield optimiser.cma.recent()


def run(optimiser, episodes, debug=False, save_every=4, **kwargs):

    for e in range(1, kwargs['epochs'] + 1):
        #monitoring...
        #wb(roc = evaluator.roc_plot())

        #each episode
        for episode in episodes:
            for batch in du.batch_iterator(episode[:-1], episode[1:], batch_size=kwargs['batch_size'], shuffle=True):
                batch_state, batch_next_state = batch
                optim(batch_state, batch_next_state)
                yield optim.cma.recent()

                
                if wb.step() % 100:
                    wb(**optim.cma.recent())
                    if debug:
                        plot_eval(model, wb)

            print("--- epoch:", e, optim.cma())

        if not e % save_every:
            wb.save(overwrite=False)
    
    print("\n\n\n\n", flush=True)
    return model

#plt.ioff()
#plt.show()

if __name__ == "__main__":

    episodes, args = initialise.initialise('sssn')
    episodes = initialise.states(episodes, shuffle=False)

    print("-- loading anomalies for live evaluation")
    a_utils = load.anomaly_utils(args)
    eval_episodes = a_utils.load_each()
    eval_episodes = a_utils.to_torch(eval_episodes)

    args.__dict__.update(OPTIM_HYPERPARAMS)

    def plot_eval(model, wb):
        if args.latent_shape == (2,):
            for a, (a_episode, n_episode) in eval_episodes.items():
                z_a = tu.to_numpy(tu.collect(model, a_episode['state']))
                z_n = tu.to_numpy(tu.collect(model, n_episode['state']))
                wb(**{a:vu.plot.plot([z_a[:,0], z_n[:,0]], [z_a[:,1], z_n[:,1]], show=False)})

    run(episodes, **args.__dict__)