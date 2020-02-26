import anomapy.load as load
from anomapy.load import anomaly_utils as au

for meta in au.meta(load.BREAKOUT, full_path=True):
    print(meta)

for a,f in au.meta_group(load.BREAKOUT):
    a_episode = au.load_anomaly(f[0])
    n_episode = au.load_normal(f[0])