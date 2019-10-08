import uuid

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

from fastai import *
from fastai.imports import *
from fastai.vision import *
from skimage.transform import resize

from feature_loss import VGGFeatureLoss

def make_channel_first(image):
    return np.rollaxis(image, 2, 0)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class Draw2DEnv(Env):
    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-float('inf'), float('inf'))
    def __init__(self, cuda=None):
        self.device = torch.device(f"cuda:{cuda}" if (cuda and torch.cuda.is_available()) else "cpu")
        self.loss = VGGFeatureLoss().to(self.device)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32)
        self.fig_id = str(uuid.uuid4())
        self.fig = plt.figure(num=self.fig_id, figsize=(2,2))
        self.ax = Axes3D(self.fig)
        self.ax.view_init(elev=0., azim=0.0) # rotate camera
        self.ax.dist = 6.2
        self.W, self.H = self.fig.canvas.get_width_height()
        self.w, self.h = 64, 64
        # self.ds_X, self.ds_y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True,
        #                                              data_home='mnist_cache', cache=True)

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
        current_img = make_channel_first(current_img)
        o = (current_img, self.goal_img) #TODO add cursor position?

        r = -self.loss(
            from_numpy(np.expand_dims(current_img,0)).to(self.device).float(),
            from_numpy(np.expand_dims(self.goal_img,0)).to(self.device).float(),
            ) # TODO save/cache goal_img features
        d = False
        i = {"is_success":False}
        return o,r,d,i

    def process_env_image(self, img):
        return make_channel_first(img) # rgb2gray(img/255)

    def reset_plot(self):
        self.ax.clear()
        self.ax.set_xlim3d(0,1)
        self.ax.set_ylim3d(0,1)
        self.ax.set_zlim3d(0,1)

    def reset(self):
        self.reset_plot()
        # i = 20 # TODO make it random, static now for debugging.
        # img_sample = self.ds_X[i].reshape(28,28)
        img_sample = 1. - (np.load('4.npy')/255.)
        self.goal_img = resize(img_sample, (self.h, self.w))
        self.goal_img = np.expand_dims(self.goal_img,0)
        self.goal_img = np.concatenate([self.goal_img] * 3, axis=0)
        return (np.zeros((3, self.h, self.w)), self.goal_img)

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def sample_action(self):
        return (np.random.rand(2)*2)-1.0
