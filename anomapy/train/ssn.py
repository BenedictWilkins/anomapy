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

from anomapy import load

from pyworld.toolkit.tools.wbutils import WB

#TODO_ENVIRONMENTS = [BREAKOUT, ENDURO, PONG, QBERT, SEAQUEST, SPACEINVADERS]]
 
env = load.BREAKOUT

train_path = load.path_train(env)
files = [file for file in fu.files(train_path, full=True)]
episode, input_shape = load.load_all(*files[:-1], **load.HYPER_PARAMETERS[env], max_size = 25000)

test, input_shape = load.load(files[-1], **load.HYPER_PARAMETERS[env])

test = torch.from_numpy(test)
episode = torch.from_numpy(episode)

latent_shape = 2
device = tu.device()
mode = optim_mode.top_n
margin = 0.2
k = 16
lr = 0.0005

model = CNet2(input_shape, latent_shape).to(device)
optim = SSTripletOptimiser(model, mode=mode, margin=margin, k=k, lr=lr)

epochs = 50
batch_size = 64

config = {'environment':env, 'latent_shape':latent_shape, 'batch_size':batch_size, 'epochs':epochs, 
          'dataset_size':episode.shape[0], 'optim_mode':mode, 'optim_margin':margin, 'optim_k':k, 'learning_rate:': lr,
          **load.HYPER_PARAMETERS[env]}

import matplotlib.pyplot as plt

wb = WB('anomapy', model, config=config)
step = 0
with wb:
    fig = None
    for e in range(epochs):
        for i, batch in enumerate(du.batch_iterator(episode[:-1], episode[1:], batch_size=batch_size, shuffle=True)):
            batch_state, batch_next_state = batch
            optim(batch_state, batch_next_state)
            if wb.step() % 100:
                wb(**optim.cma.recent())
        
        #states, distance = optim.state_state_distance(test)
        #print(states.shape)

        fig = vu.plot2D(model, test, fig=fig, marker='-')
        
        image = wb.image(fig, "latent_space")
        wb(latent_space=image)
        
        print("epoch:", e, optim.cma())
        optim.cma.reset()

        #indx = np.random.randint(0, test.shape[0], size=(14,))
        #vu.show(vu.gallery(tu.to_numpy(test[indx])), 'real')
        #vu.show(vu.gallery(tu.to_numpy(F.sigmoid(model(test[indx])))), 'recon')
        
        wb.save()


plt.ioff()
plt.show()









