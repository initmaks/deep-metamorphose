import time
import yaml
import argparse
import logging

import gym
import torch
import numpy as np
import pybullet as p
from matplotlib.image import imsave

import body_envs

logging.basicConfig(level = logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', help='filepath to .pt file with Learner Class')
    parser.add_argument('-r', help='number of rollouts', default=10)
    parser.add_argument('-c', help='Config', default=10)
    args = parser.parse_args()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    actor = torch.load(args.f, map_location=device)
    actor.device = device
    config = yaml.safe_load(open(args.c))
    start_time = time.time()
    env = gym.make(config['env_name'], gui=True, env_setup=config['env_setup'])
    after_setup_time = time.time()
    for rollout in range(int(args.r)):
        logging.info(f"rollout: {rollout}")
        o,r,d,i = env.reset()
        while not d:
            a = actor.get_action(o, True)
            o,r,d,i = env.step(a)
    logging.info(f"Sim setup completed in {after_setup_time-start_time} seconds")