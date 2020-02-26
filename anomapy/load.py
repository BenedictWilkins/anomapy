import pyworld.toolkit.tools.gymutils as gu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.visutils as vu
import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.wbutils as wbu

from pprint import pprint

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

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
#BeamRiderNoFrameskip-v4 ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE', 'LEFTFIRE'] Discrete(9)
#BreakoutNoFrameskip-v4 ['NOOP', 'FIRE', 'RIGHT', 'LEFT'] Discrete(4)
#EnduroNoFrameskip-v4 ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT', 'RIGHTFIRE', 'LEFTFIRE'] Discrete(9)
#PongNoFrameskip-v4 ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'] Discrete(6)
#QbertNoFrameskip-v4 ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN'] Discrete(6)
#SeaquestNoFrameskip-v4 ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'] Discrete(18)
#SpaceInvadersNoFrameskip-v4 ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'] Discrete(6)

# Some of the actions in each game are redundant, we need to transform the actions to include only those that are distinct.
action_transform = SimpleNamespace(**{BEAMRIDER:[], 
                                        BREAKOUT:[0,0,1,2], #
                                        ENDURO:[],
                                        PONG:[0,0,1,2,1,2],
                                        QBERT:[],
                                        SEAQUEST:[],
                                        SPACEINVADERS:[]})

#TODO change the name of this ...
ACTION_SHAPES = {k:len(np.unique(v)) for k,v in action_transform.__dict__.items()}
#ACTION_SHAPES = {BEAMRIDER:9, BREAKOUT:4, ENDURO:9, PONG:6, QBERT:6, SEAQUEST:18, SPACEINVADERS:6}


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


from collections import namedtuple, defaultdict

class anomaly_utils:

    file_meta = namedtuple('file_meta', 'file anomaly shape a_count')
    an_pair = namedtuple('an_pair', 'anomaly normal')

    def __init__(self, **kwargs):
        self.args = SimpleNamespace(**kwargs) #TODO RELIC, refactor this...
    
    def meta(self, full_path=True):
        #pprint(self.args)
        meta_f = fu.load(os.path.join(PATH_ANOMALY(self.args.env), 'meta.txt'))
        lines = [line.split() for line in meta_f if 'episode' in line]
        path = ('', PATH_ANOMALY(self.args.env))[int(full_path)]
        meta_info = [anomaly_utils.file_meta(os.path.join(path, line[0]) + '.hdf5', line[1], eval("".join(line[2:6])), int(line[6])) for line in lines]
        return meta_info

    def meta_group(self, full_path=True):
        meta = self.meta(full_path=full_path)
        a_meta = defaultdict(list)
        for m in meta:
            a_meta[m.anomaly].append(m)
        for a,m in a_meta.items():
            yield a,m

    def to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, anomaly_utils.an_pair):
            return anomaly_utils.an_pair(self.to_torch(x.anomaly), self.to_torch(x.normal))
        elif isinstance(x, dict):
            return {k:self.to_torch(v) for k,v in x.items()}
        elif isinstance(x, list):
            return [self.to_torch(v) for v in x]
        else:
            raise ValueError("invalid type: {0} cannot be converted to torch.".format(type(x)))

    def load_anomaly(self, _meta):
        print("---- loading anomaly episode: {0}".format(_meta.file))
        _, episode =  next(__load__([_meta.file], 'state', 'action', 'label'))
        episode = transform(episode, **self.args.__dict__)

        if _meta.anomaly == 'action': #take special care of this, TODO fix the dataset so this can be removed
            episode = fix_action_anomaly(_meta.file, episode, self.args)

        return episode

    def load_raw(self, _meta):
        file = _meta.file.replace('anomaly', 'raw')
        print("---- loading normal episode:  {0}".format(file))
        _, episode = next(__load__([file], 'state', 'action'))
        return transform(episode, **self.args.__dict__)

    def load_both(self, _meta):
        return anomaly_utils.an_pair(self.load_anomaly(_meta), self.load_raw(_meta))

    def load_each(self):
        return {a:self.load_both(m[0]) for a,m in self.meta_group()}

def fix_action_anomaly(file, episode, args):
    print("---- fixing action anomalies... TODO redo the dataset!")
    #assume transform has been applied - the actions/labels will be updated in the dataset at some point, at which point this will be obsolete.
    raw_file = file.replace('anomaly', 'raw')
    _, raw_episode = next(__load__([raw_file], 'action'))
    raw_actions = raw_episode['action']
    raw_actions[-1] = 0 #remove last value which is always nan
    raw_actions = remove_redundant_actions(raw_actions, args)[:, np.newaxis]
    anom_actions = episode['action']

    #if raw actions still differ after the transform, then anomaly, otherwise not!
    episode['label'] = raw_actions.squeeze() != anom_actions.squeeze()

    #print(episode['label'].shape, anom_actions.shape, raw_actions.shape)
    #print(np.concatenate((anom_actions[:100], raw_actions[:100], episode['label'][:100, np.newaxis]), axis=1))

    '''
    #visualise
    state = episode['state']
    dif = state[:-1] - state[1:]
    index = raw_actions[:-1].squeeze() == 2
    dif[index] = 0
    index = raw_actions[:-1].squeeze() == 1
    dif[index] = 0
    state = state[:-1][index]
    #vu.play(vu.transform.HWC(np.concatenate((dif, state), axis=2)))
    vu.play(vu.transform.HWC(dif), wait=100)
    '''

    return episode


def remove_redundant_actions(actions, env):
    return np.array(action_transform.__dict__[env], dtype=np.int64)[actions.astype(np.int64)]
    
#TODO move this
def transform(episode, **kwargs):
    if vu.transform.is_integer(episode['state']):
        episode['state'] = vu.transform.to_float(episode['state'])

    #remove redundant actions and convert to int64
    episode['action'][-1] = 0 #remove last value which is always nan
    episode['action'] = remove_redundant_actions(episode['action'], kwargs['env'])[:,np.newaxis]
    #episode['action'] = episode['action'].astype(np.int64)[:,np.newaxis]


    if 'colour' in kwargs and not kwargs['colour']:
        episode['state'] = vu.transform.gray(episode['state'])
        if kwargs['binary']:
            episode['state'] = vu.transform.binary(episode['state'], kwargs['binary_threshold'])
    episode['state'] = vu.transform.CHW(episode['state'])
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
    model = CNet2(args.state_shape, args.latent_shape).to(args.device)
    return __load_model__(model, args)

def load_sassn(args):
    from pyworld.toolkit.nn.CNet import CNet2
    from pyworld.toolkit.nn.MLP import MLP
    from pyworld.algorithms.optimise.TripletOptimiser import SASTripletOptimiser

    #state_model = CNet2(args.state_shape, args.latent_shape).to(args.device)
    #action_model = MLP(args.latent_shape[0] * 2 + args.action_shape[0], args.latent_shape[0], args.latent_shape[0]).to(args.device)
    
    state_model = CNet2(args.state_shape, args.latent_shape).to(args.device)
    action_model = MLP(args.latent_shape[0] * 2 + args.action_shape[0], args.latent_shape[0],
                            output_activation=F.leaky_relu).to(args.device)

    model = SASTripletOptimiser.SASModel(state_model, action_model) #the model class is part of the optimiser
    return __load_model__(model, args)

MODEL = {'auto-encoder':load_autoencoder, 
         'sssn':load_sssn,
         'sassn':load_sassn}

DEFAULT_SAVE_PATH = os.path.split(os.path.split(__file__)[0])[0] #top level module dir
assert DEFAULT_SAVE_PATH.endswith("anomapy") #YOU HAVE MOVE THE LOAD FILE OR RENAMED THE MODULE, THIS MIGHT CAUSE SOME SAVE/LOAD PROBLEMS!

def load_model(**kwargs):
    args = SimpleNamespace(**kwargs) #args is a relic, refactor this...
    try:
        if args.save_path is None:
            args.save_path = DEFAULT_SAVE_PATH
    except:
        args.save_path = DEFAULT_SAVE_PATH #save path was not found, but is required!
        print("---- warning: save_path arg was not found, using: {0}".format(args.save_path))

    if not 'force' in args.__dict__:
        args.__dict__['force'] = False
    if not 'index' in args.__dict__:
        args.__dict__['index'] = -1

    args.__dict__['run_path'] = args.save_path + "/runs/" + args.run

    #check if the run is local
    if not args.force and os.path.isdir(args.run_path):
        print("-- local data found at {0}, skipping download.".format(args.run_path))
    else:
        runs = wbu.get_runs(args.project)
        runs = {run.name:run for run in wbu.get_runs(args.project)}
        if args.run not in runs:
            raise ValueError("Invalid run: please choose from valid runs include:\n" + "\n".join(runs.keys()))
        wbu.download_files(runs[args.run], path = args.run_path, replace=args.force) #TODO force

    print("-- loading model...")
    #find all models
    config = fu.load(args.run_path + "/config.yaml")
    print("-- found config:")

    args.__dict__.update({k:config[k]['value'] for k in config if isinstance(config[k], dict) and not k.startswith("_")})

    if args.env in HYPER_PARAMETERS: #no hyperparams for this env
        args.__dict__.update(HYPER_PARAMETERS[args.env]) #if not colour these args are required for the data transform
    
    fix_old_config(args) #fix some old config problems

    #pprint(args.__dict__, indent=4)

    model, _ = MODEL[args.model](args)

    return model, args.__dict__
    