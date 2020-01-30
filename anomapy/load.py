import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu

import numpy as np
import re
import os

def get_model_files(args):
    return [file for file in fu.files(args.save_path, full=True) if "model" in file or file.endswith(".pt")] 

def load_autoencoder(args):
    import pyworld.toolkit.nn.autoencoder.AE as AE
    encoder, decoder = AE.default2D(args.input_shape, args.latent_size)
    model = AE.AE(encoder, decoder).to(args.device)

    files = fu.sort_files(get_model_files(args))
    assert len(files) > 0 #no models found?
    return fu.load(files[args.index], model=model), os.path.basename(files[args.index])

def load_sssn(args):
    raise NotImplementedError()

def load_sassn(args):
    raise NotImplementedError()


MODEL = {'auto-encoder':load_autoencoder, 
          'sssn':load_sssn,
          'sassn':load_sassn}

NO_FRAME_SKIP = "NoFrameskip-v4"
PATH = '~/Documents/repos/datasets/atari/'
LOAD_PATH  = PATH + '{0}/{1}' + NO_FRAME_SKIP + "/"

def PATH_RAW(env):
    return LOAD_PATH.format('raw', env) 

def PATH_CLEAN(env):
    return LOAD_PATH.format('clean', env)

def PATH_ANOMALY(env):
    return LOAD_PATH.format('anomaly', env)

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
H_BREAKOUT = {'binary':False, 'binary_threshold':0.2}
H_ENDURO = {'binary':False, 'binary_threshold':0.2}
H_PONG = {'binary':False, 'binary_threshold':0.5}
H_QBERT = {'binary':False, 'binary_threshold':0.3}
H_SEAQUEST = {'binary':False, 'binary_threshold':0.3}
H_SPACEINVADERS = {'binary':False, 'binary_threshold':0.2}

HYPER_PARAMETERS = {BEAMRIDER:H_BEAMRIDER, BREAKOUT:H_BREAKOUT, ENDURO:H_ENDURO, 
                    PONG:H_PONG, QBERT:H_QBERT, SEAQUEST:H_SEAQUEST, 
                    SPACEINVADERS:H_SPACEINVADERS}

def set_path(path):
    global PATH
    PATH = path

def dataset_info_raw(env):
    print("ENVIRONMENT: {0}".format(env))
    print("   PATH: {0}".format(PATH_RAW(env)))
    print("   EPISODES: {0}".format(len(files_raw(env))))

def files_raw(env):
    return fu.sort_files([file for file in fu.files(PATH_RAW(env), full=True)])

def files_clean(env):
    return fu.sort_files([file for file in fu.files(PATH_CLEAN(env), full=True)])

def files_anomaly(env):
    return fu.sort_files([file for file in fu.files(PATH_ANOMALY(env), full=True)])

def __load__(files, *args):
    for file in files:
        episode = fu.load(file)
        episode  = {arg:episode[arg][...] for arg in args}
        yield file, episode

def __load_frame_limit__(files, *args, limit=None):
    assert limit is not None
    total_frames = 0
    for file, episode in __load__(files, *args):
        yield file, episode
        total_frames += episode['state'].shape[0]
        if total_frames > limit:
            break
        
def load(path, *args, limit=None):
    files = sort_files([file for file in fu.files(path, full=True)])
    if limit is not None:
        return __load_frame_limit__(files, *args, limit=limit)
    else:
        return __load__(files, *args)   

def load_raw(env, limit=None):
    return load(PATH_RAW(env), 'state', 'action', limit=limit)

def load_clean(env, limit=None):
    return load(PATH_CLEAN(env), 'state', 'action', limit=limit)

def load_anomaly(env, anomaly=None, limit=None):
    if anomaly is not None:
        anomaly += " "# must match whole word
        path = PATH_ANOMALY(env)
        data = fu.load(path + "meta.txt")
        valid_lines = [line for line in data if anomaly in line]
        files = [path + re.findall("(episode(\([0-9]+\))?)", line)[0][0] + '.hd5f'  for line in valid_lines]
        return __load__(files, 'state', 'action', 'label')
    else:
        return load(PATH_ANOMALY(env), 'state', 'action', 'label', limit=limit)

def transform(states, binary, binary_threshold = 0.5):
    states = states.astype(np.float32) / 255. #convert to CHW format
    states = vu.transform.gray(states)
    if binary:
        states = vu.transform.binary(states, binary_threshold)
    return vu.transform.CHW(states) 


def transformer(args):
    
    if args.colour: 
        return lambda x: x[1]['state']
    else:
        args.__dict__.update(HYPER_PARAMETERS[args.env])
        return lambda x: transform(x[1]['state'], args.binary, args.binary_threshold)





'''
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
'''



    