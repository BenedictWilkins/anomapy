from functools import wraps
import numpy as np 
import torch

import pyworld.toolkit.tools.datautils as du

from pyworld.toolkit.tools.datautils.function import convolve1D_gauss
from pyworld.toolkit.tools.visutils.transform import CHW, HWC, isCHW, isHWC   

import os

import copy

SUPRESS_WARNINGS = False

def CHW_format(func):

    def _CHW(episode):
        episode['state'] = CHW(episode['state'])
        return episode
    def _HWC(episode):
        episode['state'] = HWC(episode['state'])
        return episode
    
    @wraps(func)
    def CHW_wrapper(episode, *args, **kwargs):
        if isHWC(episode['state']):
            if not SUPRESS_WARNINGS:    
                print("Warning: function \"{0}\" requires NCHW format, transforming from assumed NHWC format.".format(func.__name__))
            e, *r = func(_CHW(episode), *args, **kwargs)
            return (_HWC(e), *r)
        return func(episode)

    return CHW_wrapper

def action(episode, ratio=0.05):
    '''
        Replaces an action with a random action (that is not equal to the original).
    '''
    episode = copy.deepcopy(episode)
    action = episode['action']

    u_actions = np.arange(0, max(action))

    size = int(ratio * action.shape[0]) 
    index = np.random.randint(0, action.shape, size=size)
    
    rand = np.random.randint(1, len(u_actions), size=size)

    action[index] = (action[index] + rand) % len(u_actions)

    labels = np.zeros(action.shape[0], dtype=bool)
    labels[index] = True

    episode['action'] = action #...?

    return episode, labels

def freeze(episode, ratio=0.05, freeze_for=(4, 16)):
    '''
        Freezes an episode at successive random frames for a random number of frames given by freeze_for. 
        The result has the form (..., S_t-1, S_t, S_t, ..., S_t, S_t+1, S_t+2, ...), i.e. after a freeze, the episodes continues from the natural next state. 
        The episode will be extended by the total number of frozen frames.
        
        Anomalies are considered to be the frames during the freeze (excluding the initial frame). For example:
        
        episode = [1,2,3,4,5] -> [1,2,2,2,3,4,5]
        normal_index = [0,1,4,5,6]
        anomaly_index = [2,3]

        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples, a total of ratio * len(episode) freezes (of varying length) will be generated
            freeze_for: a tuple (a,b), a < b indicating a range for the length of a given freeze (the freeze time will be randomly generated in the given range)  
        Returns:
            episode, labels (0 = normal, 1 = anomaly)
    '''
    assert freeze_for[0] < freeze_for[1]

    episode = copy.deepcopy(episode)

    state = episode['state']
    action = episode['action']

    size = int(ratio * state.shape[0])
    a_indx = np.sort(np.random.choice(state.shape[0], size=size, replace=False))
    freeze_for = np.random.randint(low=freeze_for[0], high=freeze_for[1], size=a_indx.shape[0])
    
    #build index
    result = []
    j = 0
    for i in range(a_indx.shape[0]):
        result.extend([k for k in range(j,a_indx[i])])
        result.extend([a_indx[i]] * freeze_for[i])
        j = a_indx[i]
        
    result.extend([k for k in range(j, state.shape[0])])

    anomaly_index = np.zeros(state.shape[0], dtype=bool)
    anomaly_index[a_indx] = True

    episode['state'] = state[result]
    episode['action'] = action[result]

    assert episode['action'].shape[0] == episode['state'].shape[0] #sanity check

    anomaly_index = anomaly_index[result]
    anomaly_index[a_indx + np.cumsum(freeze_for) - freeze_for] = False #the first frame of a freeze is not an anomaly

    return episode, anomaly_index

def freeze_skip(episode, ratio=0.05, freeze_for=(4, 16)):
    '''
        Freezes an episode at successive random frames for a random number of frames given by freeze_for. 
        The result has the form (..., S_t-1, S_t, S_t, ..., S_t, S_t+n, S_t+n+1, ...), i.e. after a freeze, the episodes continues from state at the natural index.
        Anomalies are considered to be the frames during the freeze (excluding the initial frame). For example:

        episode = [1,2,3,4,5] -> [1,2,2,2,5]
        normal_index = [0,1,4]
        anomaly_index = [2,3]
        
        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples, a total of ratio * len(episode) freezes (of varying length) will be generated
            freeze_for: a tuple (a,b), a < b indicating a range for the length of a given freeze (the real lengths will be randomly generated in the given range)              
        Returns:
            episode, labels (0 = normal, 1 = anomaly)
    '''
    assert freeze_for[0] < freeze_for[1]
    
    episode = copy.deepcopy(episode)
    state = episode['state']

    size = int(ratio * state.shape[0])
    a_indx = np.sort(np.random.choice(state.shape[0], size=size, replace=False))
    freeze_for = np.random.randint(low=freeze_for[0], high=freeze_for[1], size=a_indx.shape[0])
    freeze_frames = state[a_indx]
    #freeze_frames_a = action[a_indx]
    
    anomaly_index = np.zeros(state.shape[0], dtype=bool)

    for i in range(a_indx.shape[0]):
        state[a_indx[i]+1:a_indx[i] + freeze_for[i]] = freeze_frames[i] #first frame of freeze is already filled...
        #action[a_indx[i]+1:a_indx[i] + freeze_for[i]] = freeze_frames_a[i] #??? we want to repeat the action? or leave it?

        anomaly_index[a_indx[i]+1:a_indx[i] + freeze_for[i]] = True
    
    assert episode['action'].shape[0] == episode['state'].shape[0] #sanity check

    return episode, anomaly_index
    
def split_horizontal(episode, ratio=0.05):
    return split(episode, ratio=ratio, vertical=False, horizontal=True)

def split_vertical(episode, ratio=0.05):
    return split(episode, ratio=ratio, vertical=True, horizontal=False)

@CHW_format
def split(episode, ratio=0.05, vertical=False, horizontal=True):
    '''
        Creates a split anomaly. Half (horizontal or vertical) of a state s_i is replaced with half of another state s_j. 
        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
            veritcal: vertical split?
            horizontal: horizontal split?
        Returns:
            episode, labels (0 = normal, 1 = anomaly)
    '''
    assert vertical or horizontal

    episode = copy.deepcopy(episode)
    state = episode['state']

    size = int(ratio * state.shape[0]) 
    indx = np.random.choice(state.shape[0], size=size, replace=False)
    indx = np.concatenate((indx[:,np.newaxis], np.random.randint(0, state.shape[0], indx.shape[0])[:,np.newaxis]), axis=1)

    if vertical:
        i = int(state.shape[-1]/2)
        slice = [np.s_[i:], np.s_[:i]]
        for i1, i2 in indx:
            state[i1,:,:,slice[np.random.randint(0,2)]] = state[i2,:,:,slice[np.random.randint(0,2)]]
    if horizontal:
        i = int(state.shape[-2]/2)
        slice = [np.s_[i:], np.s_[:i]]
        for i1, i2 in indx:
            state[i1,:,slice[np.random.randint(0,2)],:] = state[i2,:,slice[np.random.randint(0,2)],:]

    anom_indx = indx[:,0]
    labels = np.zeros(state.shape[0], dtype=bool)
    labels[anom_indx] = True

    return episode, labels

@CHW_format
def fill(episode, ratio=0.05, colour=None, duration=(1,5)):
    '''
        Fills the entire frame with a given colour for a number of frames. 

        Arguments:
            episode: NCHW format to generate anomalies in (a copy will be made). 
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
            colour: the fill colour
            duration: a range (a,b), a < b, used to determin how many frames to fill
        Returns:
            episode, labels (0 = normal, 1 = anomaly)
    '''
    episode = copy.deepcopy(episode)
    state = episode['state']

    if colour is not None:
        assert state.shape[1] == len(colour)
    else:
        colour = [0] * state.shape[1]
    colour = np.array(colour)[:, np.newaxis, np.newaxis]

    size = int(ratio * state.shape[0])
    a_indx = np.sort(np.random.choice(state.shape[0], size=size, replace=False))
    fill_for = np.random.randint(low=duration[0], high=duration[1], size=a_indx.shape[0])

    labels = np.zeros(state.shape[0], dtype=bool)

    for i in range(a_indx.shape[0]):
        state[a_indx[i]:a_indx[i] + fill_for[i]] = colour
        labels[a_indx[i]:a_indx[i] + fill_for[i]] = True
        
    return episode, labels

@CHW_format
def fade(episode, ratio=0.05, colour=None, sigma=1., kernel_size=5):
        
    episode = copy.deepcopy(episode)
    state = episode['state']

    if colour is not None:
        assert state.shape[1] == len(colour)
    else:
        colour = [0] * state.shape[1]
        
    colour = np.array(list(colour))


    signal = np.random.choice([0.,1.], size=state.shape[0], p=[1-ratio, ratio])
    signal = convolve1D_gauss(signal, sigma=sigma, kernel_size=kernel_size)[:, np.newaxis, np.newaxis, np.newaxis]
    colour = colour[np.newaxis, :, np.newaxis, np.newaxis]
    
    dif = colour - state

    episode['state'] = state + signal * dif

    labels = (signal[:,0,0,0] > 0.05) #maybe change this value...?
    
    return episode, labels
   
@CHW_format
def block(episode, ratio=0.1):
    '''
        Fills a (m x n) region of the state with a random colour. n and m are determined randomly.

        Arguments:
            episode: NCHW format to generate anomalies in (a copy will be made). 
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
        Returns:
            episode, normal index, anomaly index
    '''
        
    episode = copy.deepcopy(episode)
    state = episode['state']

    size = int(ratio * state.shape[0]) 
    anom_indx = np.random.choice(state.shape[0], size=size, replace=False)

    if np.max(state) > 1:
        random = lambda: np.random.randint(0, 256)
    else:
        random = lambda: np.random.uniform()
    
    for i in anom_indx:
        y1, y2 = np.random.randint(0, state.shape[-2], size=2)
        x1, x2 = np.random.randint(0, state.shape[-1], size=2)
        for j in range(state.shape[1]):
            state[i,j,min(y1, y2):max(y1, y2),min(x1, x2):max(x1, x2)] = random()
        
    labels = np.zeros(state.shape[0], dtype=bool)
    labels[anom_indx] = True
        
    return episode, labels

# ==================================================================================

FREEZE = freeze.__name__
FREEZE_SKIP = freeze_skip.__name__
SPLIT_HORIZONTAL = split_horizontal.__name__
SPLIT_VERTICAL = split_vertical.__name__
FILL = fill.__name__
BLOCK = block.__name__
ACTION = action.__name__

ANOMALIES = [ACTION, FILL, BLOCK, FREEZE, FREEZE_SKIP, SPLIT_HORIZONTAL, SPLIT_VERTICAL]

# ==================================================================================
 


if __name__ == "__main__":

    import load
    import pyworld.toolkit.tools.gymutils as gu
    import pyworld.toolkit.tools.fileutils as fu
    import pyworld.toolkit.tools.visutils as vu
    
    def show_ca(env, anomaly):
        #_, episode_clean = next(load.load_raw(env))
        _, episode_anomaly = next(load.load_anomaly(env, anomaly=anomaly))
        #episode = np.concatenate((episode_clean['state'], episode_anomaly['state']), axis=2)
        vu.play(episode_anomaly['state'], name="{0}:{1}".format(env, anomaly))
    
    def videos(env, *anomalies):
        file = '~/Documents/repos/datasets/atari/videos/{0}/{1}.mp4'
        meta_file = '~/Documents/repos/datasets/atari/videos/{0}/meta.txt'.format(env)
    
        _, episode = next(load.load_raw(env))
        #print(episode['state'].shape, episode['state'].dtype)
        
        meta_f = fu.save(meta_file, "{0}\n".format(env))
        
        for anom in anomalies:
            a_episode, labels = anom(episode['state'], ratio=0.05)
            
            meta_f.write("----------------------------------------\n")
            meta_f.write(anom.__name__ + "\n")
            meta_f.write("   anomaly prob: {0}\n".format(0.05))
            meta_f.write("   total frames: {0}\n".format(labels.shape[0]))
            meta_f.write("   normal frames: {0}\n".format(labels.shape[0] - np.sum(labels)))
            meta_f.write("   anomalous frames: {0}\n".format(np.sum(labels)))
            
            #vu.play(a_episode[np.logical_not(labels)], name=anom.__name__)
            fu.save(file.format(env, anom.__name__), a_episode, format='rgb')
            
        meta_f.close()

    def generate_anomalies(env, *anomalies, prob=0.05):
        
        len_episodes = len(load.files_raw(env))
        len_anomalies = len(anomalies)
        
        len_chunk = int(len_episodes / 10)
        print("episodes: {0}, anomalies: {1}, chunking: {2}".format(len_episodes, len_anomalies, len_chunk))
        
        _anom = [a for anom in anomalies for a in [anom] * len_chunk]
        
        meta_file = load.PATH_ANOMALY(env) + 'meta.txt'
        meta_f = fu.save(meta_file, "{0}\n".format(env))
        
        columns_ = "{0:<16} {1:<20} {2:<20} {3:<6}\n"
        
        meta_f.write("INFO: anomaly prob: {0}".format(prob))
        meta_f.write("--------------------------------------------------------------------------------\n")
        meta_f.write(columns_.format('EPISODE', 'ANOMALY', 'SHAPE', 'A_COUNT'))
        meta_f.write("--------------------------------------------------------------------------------\n")
        
        for i, fe in enumerate(load.load_raw(env)):
            file, base_episode = fe
            if i >= len(_anom):
                break
            anom = _anom[i]
            
            episode, episode['label'] = anom(base_episode, ratio=prob)
            
            e_name = os.path.splitext(os.path.basename(file))[0]
            a_count = np.sum(episode['label'])
            meta_f.write(columns_.format(e_name, anom.__name__, str(episode['state'].shape), a_count))
            #print(columns_.format(e_name, anom.__name__, str(episode['state'].shape), a_count)) 
            
            fu.save(file.replace('raw', 'anomaly'), episode)
            
        meta_f.write("--------------------------------------------------------------------------------\n")
        
        meta_f.close()
        
        #for file, episode in load.load_clean(env):
            
            #for anom in anomalies:
            #    pass #a_episode, labels = anom(episode['state'], ratio=0.05)
            
    SUPRESS_WARNINGS = True
    
    #show_ca(load.BEAMRIDER)
    import numpy as np
    anomalies = [fill, block, freeze, freeze_skip, split_horizontal, split_vertical, action]
    for env in load.ENVIRONMENTS:
        for anom in ANOMALIES:
            show_ca(env, anom)

        #_, episode = next(load.load_raw(env))
        #a_episode1, labels = split_horizontal(episode['state'], ratio=0.05)
        #a_episode2, labels = split_vertical(episode['state'], ratio=0.05)
        #a_episode = np.concatenate((a_episode1, a_episode2), axis=2)
        #vu.play(a_episode)    
        #break
        
        videos(env, *anomalies)
        generate_anomalies(env, *anomalies)