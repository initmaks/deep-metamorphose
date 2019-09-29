import numpy as np
import torch
from torch import optim

from boostup.algos.learner import Learner
from boostup.utils.core import to_tensor
from boostup.algos.td3 import core


"""

Twin Delayed Deep Deterministic Policy Gradient + 2CNN (TD3_2CNN)

"""

class TD3_CNN(Learner):
    def __init__(self,
                 device,
                 action_space,
                 gamma=0.99,
                 polyak=0.995,
                 pi_lr=1e-3,
                 q_lr=1e-3,
                 cnn_lr=1e-3,
                 policy_delay=2,
                 noise_clip=0.5,
                 noise_scale=0.1,
                 target_noise=0.2,
                 batch_size=256,
                 ac_kwargs = {},
                 ):
        super(TD3_CNN, self).__init__()
        self.device = device
        self.gamma = gamma
        self.polyak = polyak
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.target_noise = target_noise
        self.noise_scale = noise_scale
        self.act_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.batch_size = batch_size
        self.updates_made_count = 0

        ac_kwargs['action_space'] = action_space
        self.ac_main = core.ActorCriticCNN(**ac_kwargs).to(device)
        #self.ac_main = LSUVinit(self.ac_main,data)
        self.ac_target = core.ActorCriticCNN(**ac_kwargs).to(device)

        # Optimizers
        self.pi_optimizer = optim.Adam([
            *self.ac_main.policy.parameters(),
        ], lr=pi_lr)
        self.q_optimizer = optim.Adam([
            *self.ac_main.q1.parameters(),
            *self.ac_main.q2.parameters(),
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
            "q1":core.count_vars(self.ac_main.q1),
            "q2":core.count_vars(self.ac_main.q2),
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
            _, q1, q2, q1_pi = self.ac_main(x, a)
            pi_targ = self.ac_target(x2)

            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            _, q1_targ, q2_targ, _ = self.ac_target(x2, a2)
            
            # Bellman backup for Q functions, using Clipped Double-Q targets
            min_q_targ = torch.min(q1_targ, q2_targ)
            backup = (r + self.gamma*(1-d)*min_q_targ).detach()

            self.cnn_optimizer.zero_grad()

            # TD3 losses
            q1_loss = ((q1-backup)**2).mean()
            q2_loss = ((q2-backup)**2).mean()
            q_loss = q1_loss + q2_loss

            # Q-learning update
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            self.cnn_optimizer.step()

            self.updates_made_count += 1

            if self.updates_made_count % self.policy_delay == 0:
                pi_loss = -q1_pi.mean() # NOTE: q1_pi is outdated
                # Delayed policy update
                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                self.pi_optimizer.step()

                # Polyak averaging for target variables
                # Credits: https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
                for ac_target, ac_main in zip(self.ac_target.parameters(),
                                              self.ac_main.parameters()):
                    ac_target.data.copy_(ac_main.data * (1.0 - self.polyak) +
                                         ac_target.data * self.polyak)
                return {"LossPi":pi_loss.item(),
                        "LossQ":q_loss.item(),
                        "Q1Vals":q1.cpu().detach().numpy(),
                        "Q2Vals":q2.cpu().detach().numpy()}


            return {"LossQ":q_loss.item(),
                    "Q1Vals":q1.cpu().detach().numpy(),
                    "Q2Vals":q2.cpu().detach().numpy()}

    def get_action(self, obs, deterministic):
        self.eval()
        noise_scale = 0.0 if deterministic else self.noise_scale
        state, img = obs
        state = state.reshape(1,-1)
        img = np.expand_dims(img,0)
        ac_inp = (to_tensor(inp, self.device) for inp in [state, img])
        a = self.ac_main(ac_inp)
        a = a.cpu().detach().numpy()
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def checkpoint(self):
        checkpoints_dict = dict()
        # Log info about epoch
        # min_and_max
        for key in ['EpRet', 'TestEpRet','Q1Vals','Q2Vals']:
            checkpoints_dict[key] = 'mm'
        # average
        for key in ['EpLen','TestEpLen','LossPi','LossQ']:
            checkpoints_dict[key] = 'avg'
        return checkpoints_dict
