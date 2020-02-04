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

import plotly.graph_objects as go

def score_autoencoder(model, episode):
    x = torch.from_numpy(episode['state'])
    model.eval()
    z = tu.collect(model, x)
    model.train()
    y = F.binary_cross_entropy_with_logits(z, x, reduction='none')
    #print(x.shape, z.shape, y.shape)
    return  episode['label'], tu.to_numpy(y.view(y.shape[0], -1).mean(1))

def score_sssn(model, episode):
    model.eval()
    z = tu.collect(model, torch.from_numpy(episode['state']))
    model.train()

    score = ((z[:-1] - z[1:]) ** 2).sum(1) #L22 distance by default?
    label = episode['label']

    # a pair is considered anomalous if either state is labelled anomalous
    # we should pAUC instead of AUC as a measure to reduce bias caused by this?
    label = np.logical_or(label[:-1], label[1:]) 
    return label, score

MODEL_SCORE = {"auto-encoder":score_autoencoder, 
               "sssn":score_sssn}

def roc(label, score):
    assert label.shape[0] == score.shape[0]
    fpr, tpr, _ = roc_curve(label, score)
    return fpr, tpr

def histogram(label, score, bins=10, log_scale=True, title=None):

        #vu.play(vu.transform.HWC(episode['state'][episode['label']]))
        #log_axes = ['', 'log ']
        a_score = score[label]
        n_score = score[np.logical_not(label)]

        #print("-- Histogram:")
        #print("---- normal examples:", n_score.shape[0])
        #print("---- anomalous examples:", a_score.shape[0])

        return plot.histogram([a_score, n_score], legend=["anomaly", "normal"], bins=bins, log_scale=log_scale, show=False)

        #return vu.histogram([n_score, a_score], bins, alpha=0.5, log=log, title=title, xlabel='score', ylabel=log_axes[int(log)] + 'count', labels=['normal','anomaly'])

if __name__ == "__main__":

    def score_anomaly(model, episodes):
        labels = []
        scores = []
        for episode in episodes:
            label, score = MODEL_SCORE[args.model](model, episode)
            labels.append(label)
            scores.append(score)
        return np.concatenate(labels), np.concatenate(scores)
        
    def load_episodes(anomaly):
        return [load.transform(e, args) for f,e in load.load_anomaly(args.env, anomaly=anomaly)]

    def live_load_anomaly(args):
        for anomaly in video_anomaly.ANOMALIES:
            print("---- loading anomaly data: {0}".format(anomaly))
            yield anomaly, load_episodes(anomaly)

    def load_anomaly(args, anomalies=None):
        return {anomaly: load_episodes(anomaly) for anomaly in video_anomaly.ANOMALIES}

    DEFAULT_SAVE_PATH = os.getcwd() + "/runs/"
    
    def fix_old_config():
        def fix(arg, default_value):
            args.__dict__[arg] = default_value
            print("---- Warning: config doesnt contain: {0} using default value: {1}".format(arg, default_value))

        #fix input_shape
        try:
            args.state_shape = tuple(args.state_shape)
            #args.action_shape = eval(args.action_shape)
        except:
            try:

                args.state_shape = eval(args.input_shape)
                args.action_shape = (1,) #???
            except:
                fix('state_shape', (1, 210, 160))
                args.action_shape = (1,)

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
            args.latent_shape
        except:
            fix('latent_shape', args.latent_size)

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
    
    run = runs[args.run]
    if not os.path.isdir(args.save_path) or args.force:
        wbu.download_files(run, path = args.save_path) #by default the files wont be replaced if they are local
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
    histogram_path = args.save_path + "/media/histogram-{0}.png"
    roc_path = args.save_path + "/media/roc.png"
    results_path = args.save_path + "/metrics/results.txt"
    results = fu.save(results_path, "")

    print("-- successfully loaded model: {0}".format(model_file))

    roc_x = []
    roc_y = []
    roc_legend = []

    fu.save(args.save_path + "/media/temp.txt", "") #just to create the dir... TODO remove

    print("-- evaluating model")
    for anomaly, episodes in live_load_anomaly(args):
        print("---- computing score")
        label, score = score_anomaly(model, episodes) # get scores

        fig = histogram(label, score, bins=20, title=anomaly)
        fig.write_image(histogram_path.format(anomaly))

        fpr, tpr = roc(label, score)
        #fu.save(metrics_path.format(anomaly), (fpr, tpr))
        results.write("{0}: {1}\n".format(anomaly, auc(fpr, tpr)))

        roc_x.append(fpr)
        roc_y.append(tpr)
        roc_legend.append(anomaly)

    fig = plot.plot(roc_x, roc_y, legend=roc_legend)
    fig.write_image(fu.file(roc_path))

    print("-- done.")
      
    #import matplotlib.pyplot as plt












