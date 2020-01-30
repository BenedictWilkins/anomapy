from pyworld.algorithms.optimise.AEOptimiser import AEOptimiser
import pyworld.toolkit.nn.autoencoder.AE as AE

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

import argparse
from pprint import pprint

from .. import load

from pyworld.toolkit.tools.wbutils import WB


def load_mnist():
    x_train, _, x_test, _ = du.mnist()
    return torch.from_numpy(x_train), torch.from_numpy(x_test), x_train.shape[1:]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, required=True)    
    parser.add_argument("-latent_size", type=int, default=256)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-dataset_size", type=int, default=None) #use all data
    parser.add_argument("-device", type=str, default = tu.device())    
    parser.add_argument("-colour", type=bool, default=False)    
    
    args = parser.parse_args()
    
    args.__dict__['model'] = 'auto-encoder'

    def run():
        #load.load_clean(args.env)
        print("--------------------------")
        print(args.env, flush=True)
        print("--------------------------")

        #to NCHW float32 format
        print("--- loading data...")

        transform = load.transformer(args) #colour? binary? grayscale?

        pprint(args.__dict__)

        _episodes = [transform(e) for e in load.load_clean(args.env, limit=args.dataset_size)] #~100k frames
        for episode in _episodes[-1]:
            np.random.shuffle(episode)
        _episodes = [torch.from_numpy(episode) for episode in _episodes]
        
        episode_test = _episodes[-1] 
        episodes = _episodes[:-1]
        
        print("--- done.")
        
        args.__dict__['input_shape'] = episodes[0].shape[1:]
        
        encoder, decoder = AE.default2D(args.input_shape, args.latent_size)
        model = AE.AE(encoder, decoder).to(args.device)

        optim = AEOptimiser(model) #bce/wl?
    
        wb = WB('anomapy', model, id = "AE-{0}-{1}".format(args.env, fu.file_datetime()), config={arg:getattr(args, arg) for arg in vars(args)})
        
        print("--- training... ")
        
        step = 0
        with wb:
            for e in range(1, args.epochs + 1):
                
                indx = np.random.randint(0, episode_test.shape[0])
                real = tu.to_numpy(episode_test[indx])
                recon = tu.to_numpy(F.sigmoid(model(episode_test[indx].unsqueeze(0))))[0]
                image = wb.image(np.concatenate((real, recon), axis=2), "reconstruction")
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
    
    
    
    
    
    
    
    
    
