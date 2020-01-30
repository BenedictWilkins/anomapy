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

from anomapy import load

from pyworld.toolkit.tools.wbutils import WB

def image(episode, model):
    indx = np.random.randint(0, episode.shape[0])
    real = tu.to_numpy(episode[indx])
    recon = tu.to_numpy(F.sigmoid(model(episode[indx][np.newaxis,...])))[0]
    return np.concatenate((real, recon), axis=2)


def load_mnist():
    x_train, _, x_test, _ = du.mnist()
    return torch.from_numpy(x_train), torch.from_numpy(x_test), x_train.shape[1:]

#TODO_ENVIRONMENTS = [BREAKOUT, ENDURO, PONG, QBERT, SEAQUEST, SPACEINVADERS]]
 
env = load.BREAKOUT

read_path = '~/Documents/repos/datasets/atari/' + env + load.NO_FRAME_SKIP + "/"

files = [file for file in fu.files(read_path, full=True)]
episode, input_shape = load.load_all(*files[:-1], **load.HYPER_PARAMETERS[env], max_size = 1000)
test, input_shape = load.load(files[-1], **load.HYPER_PARAMETERS[env])

test = torch.from_numpy(test)
episode = torch.from_numpy(episode)

latent_shape = 256
device = tu.device()

encoder, decoder = AE.default2D(input_shape, latent_shape)
model = AE.AE(encoder, decoder).to(device)
optim = AEOptimiser(model) #bce/wl?

epochs = 1
batch_size = 64


config = {'environment':env, 'latent_shape':latent_shape, 'batch_size':batch_size, 'epochs':epochs, 'dataset_size':episode.shape[0], **load.HYPER_PARAMETERS[env]}

wb = WB('anomapy', model, config=config)

step = 0
with wb:
    for e in range(epochs):
        for i, batch in enumerate(du.batch_iterator(episode, batch_size=batch_size, shuffle=True)):
            optim(batch)
            step = wb.step()
            if step % 100:
                wb(**optim.cma.recent())

        wb(reconstruction=wb.image(image(test, model), "reconstruction"))
        
        vu.show(image(test, model))
        print("epoch:", e, optim.cma())
        #optim.cma.reset()

        '''
        indx = np.random.randint(0, test.shape[0], size=(14,))
        vu.show(vu.gallery(tu.to_numpy(test[indx])), 'real')
        vu.show(vu.gallery(tu.to_numpy(F.sigmoid(model(test[indx])))), 'recon')
        '''
        wb.save()

import wandb

encoder, decoder = AE.default2D(input_shape, latent_shape)
model2 = AE.AE(encoder, decoder)

vu.show(image(test, model2), "random")
import os
path = os.path.join(wandb.run.dir, 'model.pt')
print(path)

fu.load(path, model=model2)
#print(model2.state_dict().keys())

vu.show(image(test, model2), "loaded")

raise ValueError()