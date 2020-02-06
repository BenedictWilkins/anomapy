from . import score
from .. import video_anomaly
from .. import load

import pyworld.toolkit.tools.visutils as vu
import numpy as np

import gym

def action_meanings():
    for env in load.make_all():
        print(env.unwrapped.spec.id, env.get_action_meanings(), env.action_space)

def action_distributions():
    for env in load.ENVIRONMENTS:
        actions = np.concatenate([episode['action'][:-1] for _, episode in load.load_raw(env, types=['action'])])
        print(env, np.unique(actions, return_counts=True))

def test_split():
    for env in load.ENVIRONMENTS:
        for anom in video_anomaly.ANOMALIES:
            print("{0}:{1}".format(env, anom))
            _, episode = next(load.load_anomaly(env, anomaly=anom))
            score.subsequences(episode, score=np.zeros_like(episode['action']))

def test_action_anomaly():
    for env in load.ENVIRONMENTS:
        print(env)
        _, episode = next(load.load_raw(env))
        old_action = episode['action']
        episode, labels = video_anomaly.action(episode)
        print(np.sum(labels))
        print(np.unique(episode['action']))
    #print(np.concatenate((old_action[:,np.newaxis], episode['action'][:,np.newaxis], labels[:,np.newaxis]), axis=1)[:100])

def test_action_transform():
    env = load.PONG
    actions = np.concatenate([episode['action'][:-1] for _, episode in load.load_raw(env, types=['action'])]).astype(np.int64)
    print(actions[:100])
    print(np.array(load.action_transform.__dict__[env])[actions][:100])

#test_split()
test_action_anomaly()
#test_action_transform()
#action_distributions()
