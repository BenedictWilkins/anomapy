import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu

import numpy as np


NO_FRAME_SKIP = "NoFrameskip-v4"
LOAD_PATH  = '~/Documents/repos/datasets/atari/{0}' + NO_FRAME_SKIP + "/"

BEAMRIDER = "BeamRider"
BREAKOUT = "Breakout"
ENDURO = "Enduro"
PONG = "Pong"
QBERT = "Qbert"
SEAQUEST = "Seaquest"
SPACEINVADERS = "SpaceInvaders"

ENVIRONMENTS = [BEAMRIDER, BREAKOUT, ENDURO, PONG, QBERT, SEAQUEST, SPACEINVADERS]

#TODO thresholds...
H_BEAMRIDER = {'binary':False, 'binary_threshold':0.35}
H_BREAKOUT = {'binary':True, 'binary_threshold':0.2}
H_ENDURO = {'binary':False, 'binary_threshold':0.2}
H_PONG = {'binary':True, 'binary_threshold':0.5}
H_QBERT = {'binary':False, 'binary_threshold':0.3}
H_SEAQUEST = {'binary':False, 'binary_threshold':0.3}
H_SPACEINVADERS = {'binary':True, 'binary_threshold':0.2}

HYPER_PARAMETERS = {BEAMRIDER:H_BEAMRIDER, BREAKOUT:H_BREAKOUT, ENDURO:H_ENDURO, 
                    PONG:H_PONG, QBERT:H_QBERT, SEAQUEST:H_SEAQUEST, 
                    SPACEINVADERS:H_SPACEINVADERS}

def path_train(env):
    return LOAD_PATH.format(env)

def path_test(env):
    return LOAD_PATH.format(env) + "test/"

def load(path, binary=False, binary_threshold = 0.5):
    episode = fu.load(path)['state'][...].astype(np.float32) / 255. #convert to CHW format
    episode = vu.transform.gray(episode)
    if binary:
        episode = vu.transform.binary(episode, binary_threshold)
    episode = vu.transform.CHW(episode) 
    return episode, episode.shape[1:]

def load_all(*paths, binary=False, binary_threshold = 0.5, max_size=2000):
    size = 0
    episodes = []
    for path in paths:
        episode = load(path, binary=binary, binary_threshold=binary_threshold)[0]
        if size + episode.shape[0] > max_size:
            episode = episode[:max_size - size]
            episodes.append(episode)
            break
        size += episode.shape[0]
        episodes.append(episode)
        

    episodes = np.concatenate(episodes, axis=0)
    return episodes, episodes.shape[1:]

if __name__ == "__main__":
    path = '~/Documents/repos/datasets/atari/'


    for env, params in HYPER_PARAMETERS.items():
        episode,  _ = load(path + env + NO_FRAME_SKIP + '/episode.hd5f', **params)

        vu.play(episode)

    