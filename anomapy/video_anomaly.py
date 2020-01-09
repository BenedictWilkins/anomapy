
import numpy as np 
import torch

import pyworld.toolkit.tools.datautils as du

def freeze(episode, ratio=0.2, freeze_for=(2,4)):
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
            episode, normal index, anomaly index
    '''
    assert freeze_for[0] < freeze_for[1]

    episode = np.copy(episode)

    size = int(ratio * episode.shape[0])
    a_indx = np.sort(np.random.choice(episode.shape[0], size=size, replace=False))
    freeze_for = np.random.randint(low=freeze_for[0], high=freeze_for[1], size=a_indx.shape[0])
    
    #build index
    result = []
    j = 0
    for i in range(a_indx.shape[0]):
        result.extend([k for k in range(j,a_indx[i])])
        result.extend([a_indx[i]] * freeze_for[i])
        j = a_indx[i]
        
    result.extend([k for k in range(j, episode.shape[0])])

    anomaly_index = np.zeros(episode.shape[0], dtype=bool)
    anomaly_index[a_indx] = True

    episode = episode[result]

    #create numeric anomaly index
    anomaly_index = anomaly_index[result]
    anomaly_index[a_indx + np.cumsum(freeze_for) - freeze_for] = False #the first frame of a freeze is not an anomaly
    anomaly_index = np.arange(episode.shape[0])[anomaly_index]

    return episode, du.invert(anomaly_index, episode.shape[0]), anomaly_index

def freeze_skip(episode, ratio=0.1, freeze_for=(4, 16)):
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
            episode, normal_index, anomaly_index
    '''
    assert freeze_for[0] < freeze_for[1]
    
    episode = np.copy(episode)

    size = int(ratio * episode.shape[0])
    a_indx = np.sort(np.random.choice(episode.shape[0], size=size, replace=False))
    freeze_for = np.random.randint(low=freeze_for[0], high=freeze_for[1], size=a_indx.shape[0])
    freeze_frames = episode[a_indx]
    
    anomaly_index = np.zeros(episode.shape[0], dtype=bool)

    for i in range(a_indx.shape[0]):
        episode[a_indx[i]:a_indx[i] + freeze_for[i]] = freeze_frames[i]
        anomaly_index[a_indx[i]+1:a_indx[i] + freeze_for[i]] = True

    index = np.arange(episode.shape[0])
    normal_index = index[np.logical_not(anomaly_index)]
    anomaly_index = index[anomaly_index]

    return episode, normal_index, anomaly_index
    
def split_horizontal(episode, ratio=0.1):
    return split(episode, ratio=ratio, vertical=False, horizontal=True)

def split_vertical(episode, ratio=0.1):
    return split(episode, ratio=ratio, vertical=True, horizontal=False)

def split(episode, ratio=0.1, vertical=False, horizontal=True):
    '''
        Creates a split anomaly. Half (horizontal or vertical) of a state s_i is replaced with half of another state s_j. 
        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
            veritcal: vertical split?
            horizontal: horizontal split?
        Returns:
            the anomalous episode (which is always a copy of episode)
    '''
    assert vertical or horizontal
    
    episode = np.copy(episode)

    size = int(ratio * episode.shape[0]) 
    indx = np.random.choice(episode.shape[0], size=size, replace=False)
    indx = np.concatenate((indx[:,np.newaxis], np.random.randint(0, episode.shape[0], indx.shape[0])[:,np.newaxis]), axis=1)

    if vertical:
        i = int(episode.shape[-1]/2)
        slice = [np.s_[i:], np.s_[:i]]
        for i1, i2 in indx:
            episode[i1,:,:,slice[np.random.randint(0,2)]] = episode[i2,:,:,slice[np.random.randint(0,2)]]
    if horizontal:
        i = int(episode.shape[-2]/2)
        slice = [np.s_[i:], np.s_[:i]]
        for i1, i2 in indx:
            episode[i1,:,slice[np.random.randint(0,2)],:] = episode[i2,:,slice[np.random.randint(0,2)],:]

    anom_indx = indx[:,0]
    normal_indx = du.invert(anom_indx, episode.shape[0])
    
    return episode, normal_indx, anom_indx

def fill(episode, ratio=0.1, colour=None, fill_for=(1,3)):
    '''
        Fills the entire screen with a given colour for a number of frames. 

        Arguments:
            episode: NCHW format to generate anomalies in (a copy will be made). 
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
            colour: the fill colour
            fill_for: a range (a,b), a < b, used to determin how many frames to fill
        Returns:
            episode, normal index, anomaly index
    '''
    if colour is not None:
        assert episode.shape[1] == len(colour)
    else:
        colour = [0.] * episode.shape[1]

    episode = np.copy(episode)
    size = int(ratio * episode.shape[0]) 
    anom_indx = np.random.choice(episode.shape[0], size=size, replace=False)
    
    episode[anom_indx] = np.array(colour)[:, np.newaxis, np.newaxis]

    normal_indx = du.invert(anom_indx, episode.shape[0])
    return episode, normal_indx, anom_indx

def fade(episode, ratio=0.1, colour=None, fade_for=(4,10)):
    if colour is not None:
        assert episode.shape[1] == len(colour)
    else:
        colour = [0.] * episode.shape[1]

    episode = np.copy(episode)
    size = int(ratio * episode.shape[0]) 
    anom_indx = np.random.choice(episode.shape[0], size=size, replace=False)

    raise NotImplementedError("TODO FADING")


def block(episode, ratio=0.1):
    '''
        Fills a (m x n) region of the state with a random colour. n and m are determined randomly.

        Arguments:
            episode: NCHW format to generate anomalies in (a copy will be made). 
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
        Returns:
            episode, normal index, anomaly index
    '''
    episode = np.copy(episode)

    size = int(ratio * episode.shape[0]) 
    anom_indx = np.random.choice(episode.shape[0], size=size, replace=False)

    if np.max(episode) > 1:
        random = lambda: np.random.randint(0,256)
    else:
        random = lambda: np.random.uniform()
    
    for i in anom_indx:
        y1, y2 = np.random.randint(0, episode.shape[-2], size=2)
        x1, x2 = np.random.randint(0, episode.shape[-1], size=2)
        for j in range(episode.shape[1]):
            episode[i,j,min(y1, y2):max(y1, y2),min(x1, x2):max(x1, x2)] = random()
        
    normal_indx = du.invert(anom_indx, episode.shape[0])
    return episode, normal_indx, anom_indx

anomalies = [block, freeze, freeze_skip, split_horizontal, split_vertical]

if __name__ == "__main__":
    
    import pyworld.toolkit.tools.gymutils as gu
    import pyworld.toolkit.tools.fileutils as fu
    import pyworld.toolkit.tools.visutils as vu

    env = 'SpaceInvadersNoFrameskip-v4/'
    write_path = '~/Documents/repos/datas/atari/anomaly/' + env
    read_path = '~/Documents/repos/datasets/atari/' + env

    anomalies = [fill, block, freeze, freeze_skip,
                split_horizontal, split_vertical]

    #anomalies = [freeze_skip]

    files = [file for file in fu.files(read_path, full=True)]
    episode = fu.load(files[0])['state'][...] #convert to CHW format
    episode = vu.transform.CHW(episode)

    #episode = np.arange(10)

    for anom in anomalies:
        a_episode, n_indx, a_indx = anom(episode)
        #print(a_episode)
        #print(n_indx) 
        #print(a_indx)
        vu.play(a_episode, name = anom.__name__)