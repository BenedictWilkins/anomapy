import load
import video_anomaly

import pyworld.toolkit.tools.gymutils as gu
from pyworld.toolkit.tools import fileutils as fu
import pyworld.toolkit.tools.visutils as vu

import numpy as np
''' #meh kinda works.. horrible memory consumption...
def cat(env):
    COUNT = 16
    RT_COUNT = int(np.sqrt(COUNT))
    episodes = []
    for i, (file, episode) in enumerate(load.load_clean(env)):
        if i > COUNT:
            break
        
        print(file, episode['state'].shape)
        episodes.append(episode['state'])
    
    max_len = max(e.shape[0] for e in episodes)
    episodes = [np.pad(e, ((0, max_len - e.shape[0]), (0,0), (0,0), (0,0)), 'constant') for e in episodes]
    
    N = max_len
    H = episodes[0].shape[1]
    W = episodes[0].shape[2]
    C = episodes[0].shape[3]
    
    grid = np.zeros((N, RT_COUNT * H, RT_COUNT * W, C))
    
    for i in range(RT_COUNT):
        for j in range(RT_COUNT):
            grid[:,i*H:(i+1)*H,j*W:(j+1)*W,:] = episodes[i + j * RT_COUNT]
            
    print(grid.shape)
    vu.play(grid)
'''

if __name__ == "__main__":
    anoms = video_anomaly.ANOMALIES
    env = load.BREAKOUT
    
    
    for file, episode in load.load_anomaly(env, 'fill'):
        vu.play(episode['state'])


