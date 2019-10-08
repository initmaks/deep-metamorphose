import gym
import numpy as np
from imageio import imwrite

import envs

def make_channel_last(image):
    return np.rollaxis(image, 0, 3)

if __name__ == '__main__':
    env = gym.make('Draw2D-v0',cuda=False)
    c = 0
    for rollout_i in range(1):
        o, d = env.reset(), False
        while not d:
            c+=1
            a = np.random.rand(2)
            o,r,d,i = env.step(a)
            #d = i['is_success']
            i1,i2 = make_channel_last(o[0]), make_channel_last(o[1])
            rend_img = np.concatenate([i1,i2],axis=1)
            imwrite(f'test_imgs/{c}_i.png', rend_img)
            print(c, d)
            break