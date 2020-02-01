from pyworld.algorithms.optimise.AEOptimiser import AEOptimiser
import pyworld.toolkit.nn.autoencoder.AE as AE

import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.debugutils as debug
from pyworld.toolkit.tools.wbutils import WB

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from pprint import pprint

from .. import load
from . import initialise




if __name__ == "__main__":
    MODEL = "auto-encoder"

    episodes, args = initialise.initialise(MODEL)
    episodes, episode_test = initialise.states(episodes, test_episodes=1)

    def run():
        print("-- initialising model...")
        encoder, decoder = AE.default2D(args.state_shape, args.latent_shape)
        model = AE.AE(encoder, decoder).to(args.device)
        optim = AEOptimiser(model) #bce/wl?
        print("-- done.")

        pprint(args.__dict__)

        wb = initialise.WB(MODEL, model, args)
        #wb = WB('anomapy', model, id = "AE-{0}-{1}".format(args.env, fu.file_datetime()), config={arg:getattr(args, arg) for arg in vars(args)})
    
        print("--- training... ")

        step = 0
        with wb:
            for e in range(1, args.epochs + 1):
                
                indx = np.random.randint(0, episode_test.shape[0])
                real = tu.to_numpy(episode_test[indx])
                recon = tu.to_numpy(F.sigmoid(model(episode_test[indx].unsqueeze(0))))[0]
                image = wb.image(vu.transform.HWC(np.concatenate((real, recon), axis=2)), "reconstruction")
                wb(reconstruction=image)
                
                #each epoch
                for episode in episodes:
                    for batch in du.batch_iterator(episode, batch_size=args.batch_size):
                        optim(batch)
                        step = wb.step()
                        if step % 100:
                            wb(**optim.cma.recent())
                
                print("--- epoch:", e, optim.cma(), flush=True)
                #optim.cma.reset()
                if not e % 5:
                    print("--- saving model: epoch {0} step {1}".format(e, step))
                    wb.save(overwrite=False)
                    print("--- done", flush=True)
                    
            wb.save(overwrite=False)
            
        print("\n\n\n\n", flush=True)
        
    run()
    
    
    
    
    
    
    
    
    
