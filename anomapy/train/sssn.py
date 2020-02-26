from pyworld.algorithms.optimise.TripletOptimiser import SSTripletOptimiser, mode as optim_mode
from pyworld.toolkit.nn.CNet import CNet2

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


from types import SimpleNamespace
from pprint import pprint

from .. import load
from . import initialise
from ..evaluate import evaluate

OPTIM_HYPERPARAMS = {"optim_mode":optim_mode.all,
                     "optim_margin":0.2,
                     "optim_k":16,
                     "optim_lr":0.0005}

def default_config(env, shape, **kwargs):
    return dict(env=env, model='sssn', latent_shape=2, state_shape=shape, action_shape=None, epochs=10, 
                batch_size=64, device=tu.device(), colour=True, **OPTIM_HYPERPARAMS, **kwargs)

def run(episodes, debug=False, save_every=4, **kwargs):
    print("-- initialising model...")
    model = CNet2(kwargs['state_shape'], kwargs['latent_shape']).to(kwargs['device'])
    optim = SSTripletOptimiser(model, mode=kwargs['optim_mode'], margin=kwargs['optim_margin'], k=kwargs['optim_k'], lr=kwargs['optim_lr'])
    print("-- done.")

    wb = initialise.WB(model, **kwargs)
    print("--- training... ")

    with wb:
        for e in range(1, kwargs['epochs'] + 1):
            #monitoring...
            #wb(roc = evaluator.roc_plot())

            #each episode
            for episode in episodes:
                for batch in du.batch_iterator(episode[:-1], episode[1:], batch_size=kwargs['batch_size'], shuffle=True):
                    batch_state, batch_next_state = batch
                    optim(batch_state, batch_next_state)

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