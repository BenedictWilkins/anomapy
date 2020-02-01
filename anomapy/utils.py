import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.visutils as vu

import numpy as np

def cat_episodes(*episodes):
    ks = list(episodes[0].keys())
    result = {}
    for k in ks:
        result[k] = np.concatenate([e[k] for e in episodes])
    return result




