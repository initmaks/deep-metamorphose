import numpy as np
from gym import Env
from gym.spaces import Box

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Draw2DEnv(Env):
    metadata = {'render.modes': ['rgb_array']}
    reward_range = (-float('inf'), float('inf'))
    def __init__(self, gui=False, env_setup=None):
        self.action_space = Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32)
        self.fig = plt.figure(figsize=(2,2))
        self.ax = Axes3D(self.fig)
        self.ax.view_init(elev=0., azim=0.0) # rotate camera
        self.ax.dist = 6.2
        self.W, self.H = self.fig.canvas.get_width_height()

    def step(self, action):
        x, y, z = 1.0, action[0], action[1]

        size = 100
        color = 'black'
        marker = 'o'
        self.ax.scatter(x,y,z, c=color, marker=marker,s=size)

        self.ax.set_axis_off()
        self.fig.canvas.draw()
        byte_string = self.fig.canvas.tostring_rgb()
        current_img = np.fromstring(byte_string, dtype='uint8').reshape(self.H, self.W, 3)

        o = current_img #TODO add image
        r = 0.0 #TODO extract and compare pictures
        d = False # never perfect
        i = {}
        return o,r,d,i

    def reset_plot(self):
        self.ax.clear()
        self.ax.set_xlim3d(0,1)
        self.ax.set_ylim3d(0,1)
        self.ax.set_zlim3d(0,1)

    def reset(self):
        self.reset_plot()
        
        # self.painted_image = MNIST.random_img TODO


    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
