import gym
import numpy as np
from imageio import imwrite

import envs

if __name__ == '__main__':
    env = gym.make('Draw2D-v0',cuda=False)
    c = 0
    for rollout_i in range(2):
        o, d = env.reset(), False
        while not d:
            c+=1
            a = np.random.rand(2)
            o,r,d,i = env.step(a)
            #d = i['is_success']
        imwrite(f'test_imgs/{c}_i.png', o[0])
        print(c, d)