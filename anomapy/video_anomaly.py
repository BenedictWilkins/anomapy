
import numpy as np 
import torch
import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.fileutils as fu

import pyworld.toolkit.tools.debugutils as debug

import cv2
import matplotlib.pyplot as plt
import copy

def freeze(episode, ratio=0.1, freeze_for=(4,16)):
    '''
        Freezes an episode at successive random frames for a random number of frames given by freeze_for. 
        The result has the form (..., S_t-1, S_t, S_t, ..., S_t, S_t+1, S_t+2, ...), i.e. after a freeze, the episodes continues from the natural next state.
        
        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples, a total of ratio * len(episode) freezes (of varying length) will be generated
            freeze_for: a tuple (a,b), a < b indicating a range for the length of a given freeze (the real lengths will be randomly generated in the given range)  
        Returns:
            episode, normal index, anomaly index
    '''
    assert freeze_for[0] < freeze_for[1]

    size = int(ratio * episode.shape[0])
    a_indx = np.sort(np.random.choice(episode.shape[0], size=size, replace=False)) + 1
    freeze_for = np.random.randint(low=freeze_for[0], high=freeze_for[1], size=a_indx.shape[0])
    
    #build index
    result = []
    j = 0
    for i in range(a_indx.shape[0]):
        result.extend([k for k in range(j,a_indx[i])])
        result.extend([a_indx[i]] * freeze_for[i])
        j = a_indx[i]
        
    result.extend([k for k in range(j, episode.shape[0])])

    result = episode[result]
    
    return result, du.invert(a_indx, episode.shape[0]), a_indx, #TODO? should freeze frame also be considered anomalies?

def freeze_skip(episode, ratio=0.1, freeze_for=(4,16)):
    '''
        Freezes an episode at successive random frames for a random number of frames given by freeze_for. 
        The result has the form (..., S_t-1, S_t, S_t, ..., S_t, S_t+n, S_t+n+1, ...), i.e. after a freeze, the episodes continues from state at the natural index
        
        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples, a total of ratio * len(episode) freezes (of varying length) will be generated
            freeze_for: a tuple (a,b), a < b indicating a range for the length of a given freeze (the real lengths will be randomly generated in the given range)              
        Returns:
            episode, normal index, anomaly index
    '''
    assert freeze_for[0] < freeze_for[1]
    
    size = int(ratio * episode.shape[0])
    a_indx = np.sort(np.random.choice(episode.shape[0], size=size, replace=False))
    freeze_for = np.random.randint(low=freeze_for[0], high=freeze_for[1], size=a_indx.shape[0])
    freeze_frames = episode[a_indx]
    
    for i in range(a_indx.shape[0]):
        episode[a_indx[i]:a_indx[i] + freeze_for[i]] = freeze_frames[i]
    
    return episode, du.invert(a_indx, episode.shape[0]), a_indx #TODO? should freeze frame also be considered anomalies?
    
    
def fade(episode, ratio=0.1):
    raise NotImplementedError("TODO!")
    
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
    
    size = int(ratio * episode.shape[0]) 
    indx = np.random.choice(episode.shape[0], size=size, replace=False)
    indx = np.concatenate((indx[:,np.newaxis], np.random.randint(0, episode.shape[0], indx.shape[0])[:,np.newaxis]), axis=1)
    
    if vertical:
        iv = int(episode.shape[2]/2)
        for i1, i2 in indx:
            episode[i1,:,:,:iv] = episode[i2,:,:,:iv]
    if horizontal:
        ih = int(episode.shape[3]/2)
        for i1, i2 in indx:
            episode[i1,:,:ih,:] = episode[i2,:,:ih,:]
    
    anom_indx = indx[:,0]
    normal_indx = du.invert(anom_indx, episode.shape[0])
    
    return episode, normal_indx, anom_indx

def block(episode, ratio=0.1):
    '''
        Fills a (m x n) region of the state with a random colour. n and m are determined randomly.

        Arguments:
            episode: to generate anomalies in (a copy will be made)
            ratio: of anomalous to normal examples i.e. ratio * len(episode) anomalies will be generated
        Returns:
            episode, normal index, anomaly index
    '''
    size = int(ratio * episode.shape[0]) 
    anom_indx = np.random.choice(episode.shape[0], size=size, replace=False)
    
    for i in anom_indx:
        y1, y2 = np.random.randint(0, episode.shape[2], size=2)
        x1, x2 = np.random.randint(0, episode.shape[3], size=2)
        for j in range(episode.shape[1]):
            episode[i,j,min(y1, y2):max(y1, y2),min(x1, x2):max(x1, x2)] = np.random.uniform(low=0.0, high=1.0)
        
    normal_indx = du.invert(anom_indx, episode.shape[0])
    return episode, normal_indx, anom_indx


