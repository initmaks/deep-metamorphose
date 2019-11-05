import uuid
import sys
import os
from pathlib import Path

from PIL import Image

import torch
from torch import from_numpy
import numpy as np
from gym import Env
from gym.spaces import Box

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from skimage.transform import resize

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class Generate2D():
    def __init__(self, cuda=None):
        self.device = torch.device(f"cuda:{cuda}" if (cuda and torch.cuda.is_available()) else "cpu")
        self.action_space = Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32)
        self.fig_id = None
        self.reset_plot()

    def step(self, action):
        new_y, new_z = action[0], action[1]
        new_y, new_z = (new_y + 1.0)/2.0, (new_z + 1.0)/2.0
        x, y, z = 1.0, new_y, new_z

        size = 100
        color = 'black'
        marker = 'o'
        self.ax.scatter(x,y,z, c=color, marker=marker,s=size)

        self.ax.set_axis_off()
        self.fig.canvas.draw()
        byte_string = self.fig.canvas.tostring_rgb()
        current_img = np.fromstring(byte_string, dtype='uint8').reshape(self.H, self.W, 3)
        current_img = resize(current_img, (self.h, self.w))
        return current_img

    def reset_plot(self):
        if self.fig_id is not None: plt.close(self.fig_id)
        self.fig_id = str(uuid.uuid4())
        self.fig = plt.figure(num=self.fig_id, figsize=(2,2))
        self.ax = Axes3D(self.fig)
        self.ax.view_init(elev=0., azim=0.0) # rotate camera
        self.ax.dist = 6.2
        self.W, self.H = self.fig.canvas.get_width_height()
        self.w, self.h = 64, 64
        self.ax.set_xlim3d(0,1)
        self.ax.set_ylim3d(0,1)
        self.ax.set_zlim3d(0,1)

    def reset(self):
        self.reset_plot()


if __name__ == "__main__":
    env=Generate2D()
    action = env.action_space.sample()
    img = env.step(action)
    os.makedirs('data', exist_ok=True)
    NUM_SHARDS = 10
    NUM_STROKES_PER_SHARD = 1000
    SHARD_NUM_OFFSET = 0
    for i in range(SHARD_NUM_OFFSET, SHARD_NUM_OFFSET+NUM_SHARDS):
        print(i)
        actions = []
        strokes = []
        for idx in range(NUM_STROKES_PER_SHARD):
            if idx % 32 == 0: print(idx)
            env.reset()
            action = env.action_space.sample()
            actions.append(action)
            img = env.step(action)
            strokes.append(img*255)
        actions = np.array(actions, dtype=np.float)
        strokes = np.array(strokes, dtype=np.uint8)
        np.savez_compressed("data/episodes_{}.npz".format(i), actions=actions, strokes=strokes)
        print(f"{i}/{NUM_SHARDS} done")
