import numpy as np
import torch
from torch import nn
from gym.spaces import Box

from boostup.utils.cnns import load_cnn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes=(64,64), activation=nn.Tanh,
                 output_activation=None, output_scaler=1.):
        super(MLP, self).__init__()
        self.output_scaler = output_scaler
        layers = []
        prev_h = in_dim
        h = in_dim
        for h in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_h, h))
            layers.append(activation())
            prev_h = h
        layers.append(nn.Linear(h, hidden_sizes[-1]))
        if output_activation:
            try:
                out = output_activation(-1)
            except:
                out = output_activation()
            layers.append(out)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze() * self.output_scaler

# Credit: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_vars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x, a = None):
        pi = self.policy(x)
        if a is None:
            return pi
        else:
            q1 = self.q1(torch.cat([x, a],dim=1))
            q2 = self.q2(torch.cat([x, a],dim=1))
            q1_pi = self.q1(torch.cat([x, pi],dim=1))
            return pi, q1, q2, q1_pi

class ActorCriticCNN(nn.Module):
    def __init__(self, state_sample, hidden_sizes=(400,300), activation=nn.ReLU,
                 output_activation=nn.Tanh, action_space=None,
                 cnn_arch = None, cnn_freeze = False
                ):
        super(ActorCriticCNN, self).__init__()
        assert isinstance(action_space, Box)

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        state, img = state_sample
        state_dim = state.shape[0]

        self.cnn_freeze = cnn_freeze
        self.cnn = load_cnn(name=cnn_arch)
        net_input =  self.cnn.output_size

        self.policy = MLP(net_input * 2, list(hidden_sizes)+[act_dim], activation,
                      output_activation, output_scaler=act_limit)

        self.q1 = MLP((net_input * 2) + act_dim, list(hidden_sizes)+[1], activation,
                    output_activation=None, output_scaler=1.)

        self.q2 = MLP((net_input * 2) + act_dim, list(hidden_sizes)+[1], activation,
                    output_activation=None, output_scaler=1.)

        if cnn_freeze:
            print('CNN is frozen')
            for param in self.cnn.parameters():
                param.requires_grad = False

    def forward(self, x, a = None):
        img1, img2 = x
        f1 = self.cnn(img1)
        f2 = self.cnn(img2)
        pi = self.policy(torch.cat([f1.detach(), f2.detach()], dim=1))
        if a is None:
            return pi
        else:
            q1 = self.q1(torch.cat([f1, f2, a],dim=1))
            q2 = self.q2(torch.cat([f1, f2, a],dim=1))
            q1_pi = self.q1(torch.cat([f1.detach(), f2.detach(), pi],dim=1))
            return pi, q1, q2, q1_pi