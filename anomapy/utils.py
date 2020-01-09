import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.visutils as vu

def histogram(score, anomaly_index, bins=20, log=True, title=None):
    log_axes = ['', 'log '] 
    score_anomaly = score[anomaly_index]
    score_normal = score[du.invert(anomaly_index, shape = score.shape)]
    print("Histogram:")
    print("   normal examples:", score_normal.shape)
    print("   anomalous examples:", score_anomaly.shape)
    return vu.histogram2D([score_normal, score_anomaly], bins, alpha=0.5, log=log, title=title, xlabel='score', ylabel=log_axes[int(log)] + 'count', labels=['normal','anomaly'])
