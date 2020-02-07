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

def initialise():
    parser = argparse.ArgumentParser()
    parser.add_argument("-run", type=str, required=True)
    parser.add_argument("-device", type=str, default = tu.device()) 
    parser.add_argument("-project", type=str, default="benedict-wilkins/anomapy")   
    parser.add_argument("-save_path", type=str, default=None)
    parser.add_argument("-force", type=bool, default=False, help="will re-download from wandb cloud and force overwrite any files.")
    parser.add_argument("-index", type=int, default=-1, help="the index of the model to evaluate (-1 is the most recent model file)")
    parser.add_argument("-agg", type=str, default='sum', choices=list(model_score.aggregate.__dict__.keys()))

    args = parser.parse_args()

    model = load.load_model(args)
    args.__dict__['result_path'] = args.save_path + "/results/" + args.run

    return model, args

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
        label, score = score_anomaly(model, episodes) # get scores

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












