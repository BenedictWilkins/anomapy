from pyworld.toolkit.tools import fileutils as fu
from pyworld.toolkit.tools import visutils as vu

from .. import load

import numpy as np

SAVE_PATH = load.PATH + "/videos/{0}/example.mp4"

for env in load.ENVIRONMENTS:
    print("VIDEO FOR ENVIRONMENT: {0}".format(env))
    files = [file for file in load.files_clean(env)][:8]
    episodes = [e['state'] for f, e in load.__load__(files, "state")]
    min_length = min([e.shape[0] for e in episodes])
    min_length = min(min_length, 2000) #max 2000 frames
    episodes = [e[:min_length] for e in episodes]

    video = np.concatenate(episodes, axis=2)
    fu.save(SAVE_PATH.format(env), video, overwrite=True, format='rgb')
