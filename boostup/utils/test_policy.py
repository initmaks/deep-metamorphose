import time
import json
import joblib
import os
import os.path as osp
import numpy as np
import torch
from torch import Tensor
import boostup
from boostup.utils.core import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=1800)

#TODO multiple GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    # create the Learner class
    exp_config = json.load(open(osp.join(fpath,'config.json')))
    states = exp_config['states']

    model_weights_path = osp.join(fpath, 'simple_save'+itr)
    model = torch.load(model_weights_path).to(device)
    model.eval()

    return env, model

def get_action(model, obs, act_limit):
    state, img = obs
    state = state.reshape(1,-1)
    img = np.expand_dims(img,0)
    state, img = to_tensor(state, device), to_tensor(img, device)
    a = model((state, img))
    a = a.cpu().detach().numpy()
    return np.clip(a, -act_limit, act_limit)

def get_action_discrete(model, obs):
    state, img = obs
    state = state.reshape(1,-1)
    img = np.expand_dims(img,0)
    state, img = to_tensor(state, device), to_tensor(img, device)
    q = model((state, img))
    a_i = q.argmax()
    return env.action_space.action_list[a_i]

def make_channel_last(image):
    return np.rollaxis(image, 0, 3)

def run_policy(run_name, env, model, max_ep_len=None, num_episodes=100,
               render=False, plot=False, record=True):
    fig1 = plt.figure()
    imgs = []
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    act_limit = env.action_space.high[0]
    act_dim = env.action_space.shape[0]
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        if record:
#             im = plt.imshow(make_channel_last(o[1])[:,:,2],cmap='Greys', animated=True)
            im = plt.imshow(make_channel_last(o[1]),cmap='Greys', animated=True)
            imgs.append([im])
        if plot:
            plt.imshow(make_channel_last(o[1])[:,:,0],cmap='Greys')
            plt.show()
        if 'action_list' in env.action_space.__dict__:
            a = get_action_discrete(model, o)
        else:
            a = get_action(model, o, act_limit)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    im_ani = animation.ArtistAnimation(fig1, imgs, interval=50, repeat_delay=3000, blit=True)
    im_ani.save(f'movies/{run_name}.mp4', writer=writer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, model = load_policy(args.fpath,
                             args.itr if args.itr >=0 else 'last',
                             args.deterministic)
    run_name = args.fpath.split('/')[-2]
    run_policy(run_name, env, model, args.len, args.episodes, not(args.norender))