import numpy as np
import torch
from torch import optim

from boostup.algos.learner import Learner
from boostup.utils.core import to_tensor
from boostup.algos.ddpg import core

"""

Deep Deterministic Policy Gradient + CNN (DDPG_CNN)

"""

class DDPG_CNN(Learner):
    def __init__(self,
                 device,
                 action_space,
                 gamma=0.95,
                 polyak=0.95,
                 pi_lr=1e-3,
                 q_lr=1e-3,
                 cnn_lr=1e-3,
                 noise_scale = 0.2,
                 batch_size=256,
                 ac_kwargs = {},
                 ):
        super(DDPG_CNN, self).__init__()
        self.device = device
        self.gamma = gamma
        self.polyak = polyak
        self.noise_scale = noise_scale
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.batch_size = batch_size
        
        ac_kwargs['action_space'] = action_space
        self.ac_main = core.ActorCriticCNN(**ac_kwargs).to(device)
        self.ac_target = core.ActorCriticCNN(**ac_kwargs).to(device)

        self.pi_optimizer = optim.Adam([
            *self.ac_main.policy.parameters(),
        ], lr=pi_lr)
        self.q_optimizer = optim.Adam([
            *self.ac_main.q.parameters(),
        ], lr=q_lr)
        self.cnn_optimizer = optim.Adam([
            *self.ac_main.cnn.parameters(),
        ], lr=cnn_lr)

        self.ac_target.load_state_dict(self.ac_main.state_dict())

    def get_model(self):
        return self.ac_main

    def var_count(self,):
        var_count = {
            "cnn":core.count_vars(self.ac_main.cnn),
            "pi":core.count_vars(self.ac_main.policy),
            "q":core.count_vars(self.ac_main.q),
            "total":core.count_vars(self.ac_main)
        }
        return var_count

    def train(self,):
        self.ac_main.train()

    def eval(self,):
        self.ac_main.eval()

    def learn(self, buffer):
        batch = buffer.sample_batch(self.batch_size)
        x, x2, a, r, d = [to_tensor(batch[k], self.device) for k in
                          ['obs1', 'obs2', 'acts', 'rews', 'done']]
        _, q, q_pi = self.ac_main(x, a)
        _, _, q_pi_targ = self.ac_target(x2, a)

        # Bellman backup for Q function
        backup = (r + self.gamma*(1-d)*q_pi_targ).detach()

        # DDPG losses
        pi_loss = -q_pi.mean()
        q_loss = ((q-backup)**2).mean()

        self.cnn_optimizer.zero_grad()

        # Q-learning update
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph = True)
        self.q_optimizer.step()

        # Policy update
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        self.cnn_optimizer.step()

        # Polyak averaging for target variables
        # Credits: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
        for ac_target, ac_main in zip(self.ac_target.parameters(),
                                      self.ac_main.parameters()):
            ac_target.data.copy_(ac_main.data * (1.0 - self.polyak) +
                                 ac_target.data * self.polyak)

        return {"LossPi":pi_loss.item(),
                "LossQ":q_loss.item(),
                "QVals":q.cpu().detach().numpy()}

    def get_action(self, obs, deterministic):
        self.eval()
        noise_scale = 0.0 if deterministic else self.noise_scale
        img1, img2 = obs
        img1 = np.expand_dims(img1,0)
        img2 = np.expand_dims(img2,0)
        ac_inp = (to_tensor(inp, self.device) for inp in [img1, img2])
        a = self.ac_main(ac_inp)
        a = a.cpu().detach().numpy()
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def checkpoint(self):
        checkpoints_dict = dict()
        # Log info about epoch
        # min_and_max
        for key in ['EpRet', 'TestEpRet','QVals']:
            checkpoints_dict[key] = 'mm'
        # average
        for key in ['EpLen','TestEpLen','LossPi','LossQ']:
            checkpoints_dict[key] = 'avg'
        return checkpoints_dict