import pyworld.toolkit.tools.gymutils as gu

from anomapy import load

PATH = '~/Documents/repos/datasets/gym/{0}/'

def generate(env, policy=None, path=PATH, train_episodes=100, test_episodes=10):
    env_id = env.unwrapped.spec.id

    train_path = path.format(env_id)
    test_path = path.format(env_id) + "test/"
    
    print("GENERATING DATA FOR ENVIRONMENT: ", env_id)
    print("   PATH:", train_path)
    print("   EPISODES: TRAIN:", train_episodes, "TEST:", test_episodes)

    env = gu.wrappers.EpisodeRecordWrapper(env, train_path)
    for i in range(train_episodes):
        gu.episode(env, policy)
    env.path = test_path
    for i in range(test_episodes):
        gu.episode(env, policy)


    