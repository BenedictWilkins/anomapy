from collections import defaultdict

from pprint import pprint
import argparse
import os


from .. import utils
from .. import load
from .. import video_anomaly

import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.wbutils as wbu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.visutils as vu

from pyworld.toolkit.tools.visutils import plot

from ..train import ae

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


import types

import plotly.graph_objects as go

def __split(data, split=16):
    result = data[:(data.shape[0] // split) * split]
    return result.reshape(result.shape[0] // split, split, *result.shape[1:])

def subsequences(episode, score, split=16):
    '''
        Splits the episode and score into subsequences of a given size.
        A subsequence is considered anomalous if it contains atleast 1 anomaly.
        The episode is truncated (length // split) * split.
    '''
    state_s = __split(episode['state'], split=split)
    action_s = __split(episode['action'], split=split)
    label_s = __split(episode['label'], split=split)
    score_s = __split(score, split=split)

    label_s = np.any(label_s, axis=1).astype(int)
    print("---- subsequences: ", label_s.shape)
    print("---- anomaly: ", np.sum(label_s))
    print("---- normal:  ", -np.sum(label_s - 1))
    return {'state':state_s, 'action':action_s, 'label':label_s, 'score':score_s}

'''
def split_subsequence_load(env, anomaly, split=16):
    for _, episode in load.load_anomaly(env, anomaly=anomaly):
        print("-- {0}: {1}".format(env, anomaly))
        yield split_subsequence(episode, split=split)
'''

aggregate = types.SimpleNamespace(sum=lambda x: np.sum(x, axis=1), max=lambda x: np.max(x, axis=1))

class autoencoder:
    
    def score(model, episode, agg=aggregate.sum):
        x = torch.from_numpy(episode['state'])
        model.eval()
        z = tu.collect(model, x)
        model.train()
        y = F.binary_cross_entropy_with_logits(z, x, reduction='none')

        score = tu.to_numpy(y.reshape(y.shape[0], -1).sum(1))
        split = subsequences(episode, score)

        #print(split['score'].shape)
        #print(split['label'].shape)
        print(agg(np.array([[1,1], [1,2]])))

        score = agg(split['score'])

        label = split['label']
        assert score.shape == label.shape #sanity check

        return label, score

class sssn:
    
    def score(model, episode, agg=aggregate.sum):
        model.eval()
        z = tu.collect(model, torch.from_numpy(episode['state']))
        model.train()

        score = ((z[:-1] - z[1:]) ** 2).sum(1) #L22 distance by default?
        score = np.pad(tu.to_numpy(score), (0,1), 'constant') #assign 0 to the last value

        # this is old now... but might be useful at some point
        #label = episode['label']
        # a pair is considered anomalous if either state is labelled anomalous
        # we should pAUC instead of AUC as a measure to reduce bias caused by this?
        #label = np.logical_or(label[:-1], label[1:]) 

        split = subsequences(episode, score)
        #print(split['score'].shape)
        #print(split['label'].shape)
        #print(split['score'][:,:-1].shape)

        #ignore the last element, the addition of this score might lead to false positives...
        #TODO alternatively remove all of the problem subsequences (those that are 0 1 on a split)
        score = agg(split['score'][:,:-1]) 
        label = split['label']

        assert score.shape == label.shape #sanity check
        return label, score

class sassn:

    def score(model, episode, agg=aggregate.sum):
        states = torch.from_numpy(episode['state'])
        actions = du.onehot(episode['action'].astype(np.uint8), size=model.action_space.shape[0]) #make one-hot!
        actions = torch.from_numpy(actions)
        
        #print(states.shape, actions.shape)

        model.eval()
        score = tu.collect(model.distance, states[:-1],  actions[:-1], states[1:])
        model.train()
        
        score = np.pad(tu.to_numpy(score), (0,1), 'constant') #assign 0 to the last value
        print(agg(np.array([[1,1], [1,2]])))

        split = subsequences(episode, score)
        score = agg(split['score'][:,:-1]) 
        label = split['label']

        assert score.shape == label.shape #sanity check
        return label, score

        #raise NotImplementedError("TODO")

MODELS = {'auto-encoder':autoencoder, 'sssn':sssn, 'sassn':sassn}
