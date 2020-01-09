import torch
import torch.nn.functional as F

import numpy as np

import pyworld.toolkit.nn.autoencoder.AE as AE

import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu

from anomapy import load
from anomapy import video_anomaly as va
from anomapy import utils

import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('Qt5Agg')

def image(episode, model):
    indx = np.random.randint(0, episode.shape[0])
    real = tu.to_numpy(episode[indx])
    recon = tu.to_numpy(F.sigmoid(model(episode[indx][np.newaxis,...])))[0]
    return np.concatenate((real, recon), axis=2)

def video(episode, model):
    real = tu.to_numpy(episode[:100])
    recon = tu.to_numpy(F.sigmoid(model(episode[:100])))
    return np.concatenate((real, recon), axis=2)

def anomaly_score_bce(x, x_target):
    return F.binary_cross_entropy_with_logits(x, x_target, reduction='none').view(x.shape[0], -1).mean(1)

def anomaly_score_mse(x, x_target):
    return F.mse_loss(x, F.sigmoid(x_target), reduction='none').view(x.shape[0], -1).mean(1)

# load wandb run
CONFIG = "config.yaml"
MODEL = "model.pt"
HISTOGRAM = "histogram/"

#path = "~/Documents/repos/anomapy/wandb/Beamrider-run-20200102_141012-20200102141012/"
path = "~/Documents/repos/anomapy/wandb/Pong-run-20200101_211510-20200101211510/"

config = fu.load(path + CONFIG)

env = config['environment']['value']
latent_shape = config['latent_shape']['value']
binary_threshold = config['binary_threshold']['value']
binary = config['binary']['value']
batch_size = config['batch_size']['value']

device = tu.device()

# load test episode 
test_path = load.path_test(env)
#train_path = load.path_train(env)

load_files = [file for file in fu.files(test_path, full=True)]
episode, input_shape = load.load(load_files[0], binary=binary, binary_threshold=binary_threshold)

episode = torch.from_numpy(episode)

# load model
encoder, decoder = AE.default2D(input_shape, latent_shape)
model = AE.AE(encoder, decoder).to(device)
fu.load(path + MODEL, model=model)
model.eval()

#vu.play(video(episode, model))

# create anomalies
for anomaly in va.anomalies:
    # generate anomaly
    a_episode, normal_index, anomaly_index = anomaly(episode)
    a_episode = torch.from_numpy(a_episode)

    recon = np.concatenate([tu.to_numpy(F.sigmoid(model(batch))) for batch in du.batch_iterator(a_episode, batch_size=batch_size)], axis=0)    
    recon = torch.from_numpy(recon)

    score_bce = tu.to_numpy(anomaly_score_bce(recon, a_episode))

    fig = utils.histogram(score_bce, anomaly_index, title=anomaly.__name__ + "-bce")
    
    image = vu.figtoimage(fig)

    fu.save(path + HISTOGRAM + anomaly.__name__ + ".png", image)

plt.ioff()
plt.show()