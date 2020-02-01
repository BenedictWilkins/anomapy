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

from pprint import pprint

from .. import load
from . import initialise

if __name__ == "__main__":
    MODEL = "sssn"

    episodes, args = initialise.initialise(MODEL)
    episodes, episode_test = initialise.states(episodes, test_episodes=1, shuffle=False)

    OPTIM_HYPERPARAMS = {"optim_mode":optim_mode.all,
                         "optim_margin":0.2,
                         "optim_k":16,
                         "optim_lr":0.0005}

    args.__dict__.update(OPTIM_HYPERPARAMS)

    def run():

        print("-- initialising model...")
        model = CNet2(args.state_shape, args.latent_shape).to(args.device)
        optim = SSTripletOptimiser(model, mode=args.optim_mode, margin=args.optim_margin, k=args.optim_k, lr=args.optim_lr)
        print("-- done.")

        pprint(args.__dict__)

        wb = initialise.WB(MODEL, model, args)

        print("--- training... ")
        fig = None

        step = 0
        with wb:
            for e in range(1, args.epochs + 1):
                if args.latent_shape == 2:
                    fig = vu.plot2D(model, episode_test, fig=fig, marker='-', draw=False)
                    image = wb.image(fig, "latent_space")
                    wb(latent_space=image)

                #each episode
                for episode in episodes:
                    for batch in du.batch_iterator(episode[:-1], episode[1:], batch_size=args.batch_size, shuffle=True):
                        batch_state, batch_next_state = batch
                        optim(batch_state, batch_next_state)

                        step = wb.step()
                        if step % 100:
                            wb(**optim.cma.recent())
                    
                    print("--- epoch:", e, optim.cma())
                #optim.cma.reset()

                if not e % 2:
                    print("--- saving model: epoch {0} step {1}".format(e, step))
                    wb.save(overwrite=False)
                    print("--- done")
                    
                    
            wb.save(overwrite=False)
            
        print("\n\n\n\n", flush=True)

    run()

#plt.ioff()
#plt.show()









