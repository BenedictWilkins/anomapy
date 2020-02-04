from pyworld.toolkit.tools.visutils import plot

from .. import load

if __name__ == "__main__":
    _, episode = next(load.load_anomaly(load.BREAKOUT))
    states = episode['state']
    labels = episode['label']
    print(states.shape)
    plot.play(states)


    
