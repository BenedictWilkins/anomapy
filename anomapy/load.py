import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.wbutils as wbu

from pprint import pprint

from types import SimpleNamespace

import numpy as np
import re
import os
import gym


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

def make_all():
    for env in ENVIRONMENTS:
        yield gym.make(env + NO_FRAME_SKIP)

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

# --------------------- ACTION INFORMATION ----------------------- #
# Some of the actions in each game are redundant, we need to transform the actions to include only those that are distinct.
action_transform = SimpleNamespace(**{BEAMRIDER:[], 
                                    BREAKOUT:[],
                                    ENDURO:[],
                                    PONG:[0,0,1,2,1,2],
                                    QBERT:[],
                                    SEAQUEST:[],
                                    SPACEINVADERS:[]})

#TODO change the name of this ...
#ACTION_SHAPES = {k:len(np.unique(v)) for k,v in action_transform.__dict__.items()}
ACTION_SHAPES = {BEAMRIDER:9, BREAKOUT:4, ENDURO:9, PONG:6, QBERT:6, SEAQUEST:18, SPACEINVADERS:6}


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
    files = fu.sort_files([file for file in fu.files(path, full=True)])
    if limit is not None:
        return __load_frame_limit__(files, *args, limit=limit)
    else:
        return __load__(files, *args)   

def load_raw(env, limit=None, types=['state', 'action']):
    return load(PATH_RAW(env), *types, limit=limit)

def load_clean(env, limit=None, types=['state', 'action']):
    return load(PATH_CLEAN(env), *types, limit=limit)

def load_anomaly(env, anomaly=None, limit=None):
    if anomaly is not None:
        anomaly += " "# must match whole word
        path = PATH_ANOMALY(env)
        data = fu.load(path + "meta.txt")
        valid_lines = [line for line in data if anomaly in line]
        files = [path + re.findall("(episode(\([0-9]+\))?)", line)[0][0] + '.hdf5'  for line in valid_lines]
        return __load__(files, 'state', 'action', 'label')
    else:
        return load(PATH_ANOMALY(env), 'state', 'action', 'label', limit=limit)

def transform(episode, args):
    states = episode['state'].astype(np.float32) / 255. #convert to CHW format

    #remove redundant actions and convert to int64
    episode['action'][-1] = 0 #remove last value which is always nan
    #episode['action'] = np.array(action_transform.__dict__[args.env], dtype=np.int64)[episode['action'].astype(np.int64)][:,np.newaxis]
    episode['action'] = episode['action'].astype(np.int64)[:,np.newaxis]

    if not args.colour:
        states = vu.transform.gray(states)
        if args.binary:
            states = vu.transform.binary(states, args.binary_threshold)
    episode['state'] = vu.transform.CHW(states)
    return episode 


# -------------- MODEL LOADING --------------- #

def fix_old_config(args):
    def fix(arg, default_value):
        args.__dict__[arg] = default_value
        print("---- Warning: config doesnt contain: {0} using default value: {1}".format(arg, default_value))

    #fix input_shape
    try:
        args.state_shape = tuple(args.state_shape)
    except:
        try:
            args.state_shape = eval(args.input_shape)
        except:
            fix('state_shape', (1, 210, 160))
    
    try:
        args.action_shape = tuple(args.action_shape)
    except:
        args.action_shape = None

    #fix colour
    try:
        args.colour
    except:
        fix('colour', False)
    
    #fix model
    try:
        args.model
    except:
        fix('model', "auto-encoder")

    try:
        if isinstance(args.latent_shape, int):
            args.latent_shape = (args.latent_shape, )
        else:
            args.latent_shape = tuple(args.latent_shape)
    except:
        fix('latent_shape', args.latent_size)

def get_model_files(args):
    return [file for file in fu.files(args.run_path, full=True) if "model" in file or file.endswith(".pt")] 

def __load_model__(model, args):
    files = fu.sort_files(get_model_files(args))
    assert len(files) > 0 #no models found?
    fu.load(files[args.index], model=model) #does not return anything to prevent a weird error...
    return model, os.path.basename(files[args.index])

def load_autoencoder(args):
    import pyworld.toolkit.nn.autoencoder.AE as AE
    encoder, decoder = AE.default2D(args.state_shape, args.latent_shape)
    model = AE.AE(encoder, decoder).to(args.device)
    return __load_model__(model, args)

def load_sssn(args):
    from pyworld.toolkit.nn.CNet import CNet2
    print(args.state_shape)
    model = CNet2(args.state_shape, args.latent_shape).to(args.device)
    return __load_model__(model, args)

def load_sassn(args):
    from pyworld.toolkit.nn.CNet import CNet2
    from pyworld.toolkit.nn.MLP import MLP
    from pyworld.algorithms.optimise.TripletOptimiser import SASTripletOptimiser

    state_model = CNet2(args.state_shape, args.latent_shape).to(args.device)
    action_model = MLP(args.latent_shape[0] * 2 + args.action_shape[0], args.latent_shape[0], args.latent_shape[0]).to(args.device)
    model = SASTripletOptimiser.SASModel(state_model, action_model) #the model class is part of the optimiser
    return __load_model__(model, args)

MODEL = {'auto-encoder':load_autoencoder, 
         'sssn':load_sssn,
         'sassn':load_sassn}

def load_model(args):
    
    try:
        if args.save_path is None:
            args.save_path = os.getcwd()
    except:
        print("---- warning: save_path arg was not found, using: {0}".format(str(os.getcwd())))
        args.save_path = os.getcwd() #save path was not found, but is required!


    if not 'force' in args.__dict__:
        args.__dict__['force'] = False
    if not 'index' in args.__dict__:
        args.__dict__['index'] = -1

    args.__dict__['run_path'] = args.save_path + "/runs/" + args.run

    runs = wbu.get_runs(args.project)
    runs = {run.name:run for run in wbu.get_runs(args.project)}
    if args.run not in runs:
        raise ValueError("Invalid run: please choose from valid runs include:\n" + "\n".join(runs.keys()))
    
    run = runs[args.run]
    if not os.path.isdir(args.run_path) or args.force:
        wbu.download_files(run, path = args.run_path) #by default the files wont be replaced if they are local
    else:
        print("-- local data found at {0}, skipping download.".format(args.run_path))
    
    print("-- loading model...")

    #find all models
    config = fu.load(args.run_path + "/config.yaml")
    print("-- found config:")

    args.__dict__.update({k:config[k]['value'] for k in config if isinstance(config[k], dict) and not k.startswith("_")})
    args.__dict__.update(HYPER_PARAMETERS[args.env]) #if not colour these args are required for the data transform

    fix_old_config(args) #fix some old config problems

    pprint(args.__dict__, indent=4)

    model, _ = MODEL[args.model](args)

    return model
    