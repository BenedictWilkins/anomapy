from pyworld.algorithms.optimise.TripletOptimiser import SASTripletOptimiser, mode as optim_mode
from pyworld.toolkit.nn.CNet import CNet2
from pyworld.toolkit.nn.MLP import MLP

import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.debugutils as debug
import pyworld.toolkit.tools.wbutils as wbu

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from pprint import pprint

from .. import load
from .. import video_anomaly
from . import initialise
from ..evaluate import score, evaluate

if __name__ == "__main__":

    wbu.dryrun(True)

    MODEL = "sassn"

    episodes, args = initialise.initialise(MODEL)
    episodes = initialise.states_actions(episodes)

    print("-- loading anomalies for live evaluation")
    a_utils = load.anomaly_utils(args)
    meta = a_utils.meta()[0]
    eval_episodes = a_utils.to_torch({meta.anomaly:a_utils.load_both(meta)})
    #eval_episodes = a_utils.to_torch(a_utils.load_each())

    def plot_fancy(model, wb):
        if args.latent_shape == (2,):

            def funk(episode):
                x_ = tu.collect(model.state, episode['state'])
                if len(episode['action'].squeeze().shape) == 1:
                    episode['action'] = torch.from_numpy(du.onehot(episode['action'], args.action_shape[0])) #one-hotify
                print(x_.shape, episode['action'].shape)
                x = torch.cat((x_[:-1], x_[1:], episode['action'][:-1]), 1).to(model.device)
                z = tu.collect(model.action, x).sum(1)
                return tu.to_numpy(x), tu.to_numpy(z)
                
            fig = vu.plot.subplot(rows=2)

            for name, (a_episode, n_episode) in eval_episodes.items():
                # | x1 | x2 | a | is used by the optimiser!
                x, z = funk(n_episode)
                x = tu.to_numpy(x)
                z = tu.to_numpy(z)
                print(x.shape)
                print(z.shape)
                fig = vu.plot.plot_coloured(x[:,0], x[:,1], z, fig=fig, show=False, row=1, col=1)
                fig = vu.plot.histogram(z, bins=100, fig=fig, show=False, row=2, col=1)
                fig.show()


    def plot_eval(model, wb=None):
        if args.latent_shape == (2,):
            plots = []
            for a, (a_episode, n_episode) in eval_episodes.items():
                z_a = tu.to_numpy(tu.collect(model, a_episode['state']))
                z_n = tu.to_numpy(tu.collect(model, n_episode['state']))
                plot = vu.plot.plot([z_a[:,0], z_n[:,0]], [z_a[:,1], z_n[:,1]], show=False)
                if wb is not None:
                    wb(**{a:plot})
                else:
                    plots.append(plot)
            if wb is None:
                vu.plot.subplot(*plots)

    OPTIM_HYPERPARAMS = {"optim_mode":optim_mode.all,
                         "optim_margin":0.2,
                         "optim_k":16,
                         "optim_lr":0.0005}

    args.__dict__.update(OPTIM_HYPERPARAMS)

    def run():

        print("-- initialising model...")

        state_model = CNet2(args.state_shape, args.latent_shape).to(args.device)
        action_model = MLP(args.latent_shape[0] * 2 + args.action_shape[0], args.latent_shape[0]).to(args.device)

        optim = SASTripletOptimiser(state_model, action_model, mode=args.optim_mode, margin=args.optim_margin, k=args.optim_k, lr=args.optim_lr)
        print("-- done.")

        pprint(args.__dict__)

        wb = initialise.WB(MODEL, optim.model, args)
   
        print("--- training... ")
        fig = None

        step = 0
        with wb:
            for e in range(1, args.epochs + 1):
                #if args.latent_shape == 2:
                #    fig = vu.plot2D(model, episode_test, fig=fig, marker='-', draw=False)
                #    image = wb.image(fig, "latent_space")
                #    wb(latent_space=image)
                #s = score.MODELS[args.model].score_raw(optim.model, {'state':episode_test[0][0], 'action':episode_test[0][1]})
                #vu.plot.to_numpy(vu.plot.histogram(s, bins=50, show=False))
                #wb(distance=wb.histogram(s))
                

                #wb(action_anomaly_histogram = wb.image(gallery, "action_anomaly_histogram")) 

                #each episode
                for state, action in episodes:
                    for batch in du.batch_iterator(state[:-1], state[1:], action[:-1], batch_size=args.batch_size, shuffle=True):
                        batch_state, batch_next_state, batch_action = batch
                        optim(batch_state, batch_action, batch_next_state)

                        step = wb.step()
                        if step % 100:
                            wb(**optim.cma.recent())
                            
                            #plot_eval(optim.model.state, wb)
                    
                    print("--- epoch:", e, optim.cma())
                    
                plot_fancy(optim.model, wb)

                if not e % 2:
                    print("--- saving model: epoch {0} step {1}".format(e, step))
                    wb.save(overwrite=False)
                    print("--- done")
                    
                    
            wb.save(overwrite=False)
            
        print("\n\n\n\n", flush=True)

    run()

#plt.ioff()
#plt.show()


#python -m anomapy.train.sassn -env Breakout -batch_size 64 -dataset_size 10000 -latent_shape 2







