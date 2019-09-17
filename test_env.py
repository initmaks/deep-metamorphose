import gym
import numpy as np
from imageio import imwrite

import envs

if __name__ == '__main__':
    env = gym.make('Draw2D-v0')
    c = 0
    o, d = env.reset(), False
    while not d:
        c+=1
        a = np.random.rand(2)
        o,r,d,i = env.step(a)
    imwrite(f'test_imgs/{c}.png', o)
    print(c, d)