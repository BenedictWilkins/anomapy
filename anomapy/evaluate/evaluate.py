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

from . import score as model_score

from ..train import ae

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import plotly.graph_objects as go


#TODO REFACTOR
class LiveTracker:

    def __init__(self, model, args):
        self.model = model
        self.anomaly_episodes = {}
        self.anomaly_score = model_score.MODELS[args.model].score
        
        #special treatment for action anomalies that need to be fixed....
        print("---- loading anomaly {0} for live model evaluation".format(video_anomaly.ACTION))
        file, anomaly_episode = next(load.load_anomaly(args.env, video_anomaly.ACTION))
        anomaly_episode = load.transform(anomaly_episode, args)
        self.anomaly_episodes[video_anomaly.ACTION] = load.fix_action_anomaly(file, anomaly_episode, args)
        
        for anomaly in [a for a in video_anomaly.ANOMALIES if a != 'action']:
            print("---- loading anomaly {0} for live model evaluation".format(anomaly))
            _, anomaly_episode = next(load.load_anomaly(args.env, anomaly))
            self.anomaly_episodes[anomaly] = load.transform(anomaly_episode, args)
    '''
    def histogram_gallery(self):
        hists = []
        for anomaly_episode in self.anomaly_episodes:
            a_hist = vu.plot.to_numpy(histogram(*), scale=0.25)
            hists.append(a_hist)
        return vu.gallery(np.array(hists), 3)
    '''
    def roc_plot(self):
        fprs = []
        tprs = []
        legend = []
        for anomaly, episode in self.anomaly_episodes.items():
            fpr, tpr, _ = roc_curve(*self.anomaly_score(self.model, episode))
            fprs.append(fpr)
            tprs.append(tpr)
            legend.append(anomaly)
        return vu.plot.plot(fprs, tprs, legend=legend, show=False)

def score_anomaly(model, episodes, args):
    labels = []
    scores = []
    mscore = model_score.MODELS[args.model]
    aggregate = model_score.aggregate.__dict__[args.agg]
    print("USING AGGREGATE: {0}".format(args.agg))
    for episode in episodes:
        label, score = mscore.score(model, episode, agg=aggregate)
        labels.append(label)
        scores.append(score)
    return np.concatenate(labels), np.concatenate(scores)
    
def load_episodes(anomaly, args):
    return [load.transform(e, args) for f,e in load.load_anomaly(args.env, anomaly=anomaly)]

def live_load_anomaly(args):
    for anomaly in video_anomaly.ANOMALIES:
        print("---- loading anomaly data: {0}".format(anomaly))
        yield anomaly, load_episodes(anomaly, args)

def load_anomaly(args, anomalies=None):
    return {anomaly: load_episodes(anomaly, args) for anomaly in video_anomaly.ANOMALIES}


# ------------------------------------------------------------------
# --------------------------- INITIALISE ---------------------------
# ------------------------------------------------------------------
# The following functions will load a previously trained model from 
# the wandb server (or locally if it exists).
# TODO move this somewhere more general (e.g. wandb utils)

def __initialise_cmd(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-run", type=str, required=True)
    parser.add_argument("-device", type=str, default=kwargs['device']) 
    parser.add_argument("-project", type=str, default=kwargs['project'])   
    parser.add_argument("-save_path", type=str, default=kwargs['save_path'])
    parser.add_argument("-force", type=bool, default=kwargs['force'], help="will re-download from wandb cloud and force overwrite any files.")
    parser.add_argument("-index", type=int, default=kwargs['index'], help="the index of the model to evaluate (-1 is the most recent model file)")
    parser.add_argument("-agg", type=str, default=kwargs['agg'], choices=list(model_score.aggregate.__dict__.keys()))

    args = parser.parse_args()
    return load.load_model(**args.__dict__)

def initialise(run=None, device=tu.device(), project="benedict-wilkins/anomapy", save_path=None, force=False, index=-1, agg='sum'):
    kwargs = dict(run=run, device=device, project=project, save_path=save_path, force=force, index=index, agg=agg)
    if run is None:
        model, kwargs = __initialise_cmd(**kwargs) #initialise from command line arguments
    else:
        model, kwargs = load.load_model(**kwargs) #initialise from kwargs

    kwargs['result_path'] = kwargs['save_path'] + "/results/" + kwargs['run']

    return model, kwargs

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


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

    model, args  = initialise()
        
    histogram_path = args.result_path + "/media/histogram-{0}.png"
    roc_path = args.result_path + "/media/roc.png"
    results_path = args.result_path + "/metrics/results.txt"
    results = fu.save(results_path, "")

    #make the directories...
    fu.mkdir(args.result_path + "/media")
    fu.mkdir(args.result_path + "/metrics")

    print("-- successfully loaded model: {0}".format(model_file))

    roc_x = []
    roc_y = []
    roc_legend = []

    print("-- evaluating model")
    for anomaly, episodes in live_load_anomaly(args):
        print("---- computing score")
        label, score = score_anomaly(model, episodes, args) # get scores

        fig = histogram(label, score, bins=50, title=anomaly)
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












