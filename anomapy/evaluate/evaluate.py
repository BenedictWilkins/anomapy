from collections import defaultdict

from pprint import pprint
import argparse
import os


from .. import load
from .. import video_anomaly

import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.fileutils as fu
import pyworld.toolkit.tools.wbutils as wbu

if __name__ == "__main__":

    DEFAULT_SAVE_PATH = os.getcwd() + "/runs/"
    
    def fix_old_config():
        def fix(arg, default_value):
            args.__dict__[arg] = default_value
            print("---- Warning: config doesnt contain: {0} using default value: {1}".format(arg, default_value))

        #fix input_shape
        try:
            args.input_shape = eval(args.input_shape)
        except:
            fix('input_shape', (1, 210, 160))

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


    def load_episodes(anomaly):
        transform = load.transformer(args)
        return [transform(e) for e in load.load_anomaly(args.env, anomaly=anomaly)]

    parser = argparse.ArgumentParser()
    parser.add_argument("-run", type=str, required=True)
    parser.add_argument("-device", type=str, default = tu.device()) 
    parser.add_argument("-project", type=str, default="benedict-wilkins/anomapy")   
    parser.add_argument("-save_path", type=str, default=None)
    parser.add_argument("-force", type=bool, default=False, help="will re-download from wandb cloud and force overwrite any files.")
    parser.add_argument("-index", type=int, default=-1, help="the index of the model to evaluate (-1 is the most recent model file)")

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = DEFAULT_SAVE_PATH + args.run

    runs = wbu.get_runs(args.project)
    runs = {run.name:run for run in wbu.get_runs(args.project)}
    if args.run not in runs:
        raise ValueError("Invalid run: please choose from valid runs include:\n" + "\n".join(runs.keys()))

    if not os.path.isdir(args.save_path) or args.force:
        wbu.download_files(runs[args.run], path = args.save_path) #by default the files wont be replaced if they are local
    else:
        print("-- local data found at {0}, skipping download.".format(args.save_path))
    
    print("-- loading model...")

    #find all models
    config = fu.load(args.save_path + "/config.yaml")
    print("-- found config:")

    args.__dict__.update({k:config[k]['value'] for k in config if isinstance(config[k], dict) and not k.startswith("_")})
    args.__dict__.update(load.HYPER_PARAMETERS[args.env]) #if not colour these args are required for the data transform

    fix_old_config() #fix some old config problems

    pprint(args.__dict__, indent=4)

    model, model_file = load.MODEL[args.model](args)

    print("-- successfully loaded model: {0}".format(model_file))

    print("-- evaluating model")
    for anomaly in video_anomaly.ANOMALIES:
        print("---- loading anomaly data: {0}".format(anomaly))
        episodes = load_episodes(anomaly)
        #run the various tests on this data 
        





