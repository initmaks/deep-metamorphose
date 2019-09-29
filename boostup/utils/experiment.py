import time
import copy

import gym
import numpy as np
import torch
from torch import from_numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import boostup
from boostup.utils.replay_buffer import *
from boostup.utils.logx import EpochLogger

def make_channel_last(image):
    return np.rollaxis(image, 0, 3)

class Experiment():
    def __init__(self, learner_fn, config):
        if not config['seed']: config['seed'] = np.random.randint(100000)
        self.logger = EpochLogger(**config['logger_kwargs'])
        self.logger.save_config(config)
        
        self.config = config
        device = torch.device(f"cuda:{self.config['cuda']}"
                              if torch.cuda.is_available()
                              and self.config['cuda'] is not None else "cpu")
        self.config['learner_kwargs']['device'] = device

        self.env = gym.make(config['env_name'])
        self.test_env = gym.make(config['env_name'])

        self.set_seed(config['seed'])
        
        self.sample_obs = self.env.reset()

        self.learner = self.init_learner(learner_fn)
        self.logger.cloud_logger.watch(self.learner.get_model())

        self.epoch = 0
        self.steps_count = 0

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(seed)
        # https://pytorch.org/docs/master/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def save(self, step):
        save_dict = {'env': self.env,
                     'config': self.config}
        self.logger.save_state(save_dict, self.learner, step)
    
    def load(self,):
        # TODO
        pass 

    def apply_wrappers(self):
        if not self.config['wrappers']: return
        for wrapper_name in self.config['wrappers']:
            wrapper = eval(wrapper_name)
            if 'Fetch' in wrapper_name:
                self.env = wrapper(self.env, self.states)
                self.test_env = wrapper(self.test_env, self.states)
            else:
                self.env = wrapper(self.env)
                self.test_env = wrapper(self.test_env)

    def setup_buffer(self):
        obs = self.env.reset()
        act_dim = self.env.action_space.shape[0]
        obs_dim = obs.shape[0]
        self.buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=self.config['replay_size'])

    def init_learner(self, learner_fn):
        self.config['learner_kwargs']['ac_kwargs']['state_sample'] = self.sample_obs
        self.config['learner_kwargs']['action_space'] = self.env.action_space
        learner = learner_fn(**self.config['learner_kwargs'])
        self.logger.log(f'\nNumber of parameters: {learner.var_count()}\n')
        return learner

    def rollout(self):
        o, d = self.env.reset(), False
        ep_ret, ep_len = 0, 0
        while not (d or (ep_len == self.config['max_ep_len'])):
            if self.steps_count >= self.config['exploration_steps']:
                a = self.learner.get_action(o, deterministic = False)
            else:
                a = self.env.unwrapped.sample_action()
            # Step the env
            o2, r, d, i = self.env.step(a)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.config['max_ep_len'] else d
            if i['is_success']: d = True

            # Store experience to replay buffer
            self.buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2
            self.steps_count += 1
            if self.steps_count % self.config['env_steps_per_epoch'] == 0: break
        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
        return ep_len

    def improve(self, episode_len):
        self.learner.train()
        for _ in range(episode_len):
            # buffer or batch sample
            metrics = self.learner.learn(self.buffer)
            self.logger.store(**metrics)

    def zero_epoch_eval(self, n = 10):
        self.evaluate(n=30)
        self.logger.log_tabular('Epoch', self.epoch, step=0)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True, step=0)
        self.logger.log_tabular('TestEpLen', average_only=True, step=0)
        self.logger.dump_tabular()
        self.logger.first_row = True
        self.logger.log_headers = []

    def evaluate(self, n=10, runs_to_record = 3): #test - fully deterministic
        self.learner.eval()
        img_list = []; fig1 = plt.figure()
        o, d = self.test_env.reset(), False
        for j in range(n):
            o, d = self.test_env.reset(), False
            ep_ret, ep_len = 0, 0
            while not(d or (ep_len == self.config['max_ep_len'])):
                # Take deterministic actions at test time (noise_scale=0)
                a = self.learner.get_action(o, deterministic = True)
                o, r, d, i = self.test_env.step(a)
                if i['is_success']: d = True
                ep_ret += r
                ep_len += 1
                if j < runs_to_record:
                    i1,i2 = make_channel_last(o[0]), make_channel_last(o[1])
                    rend_img = np.concatenate([i1,i2],axis=1)/255.
                    img_list.append([plt.imshow(rend_img, animated=True)])
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        if img_list:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, bitrate=1800)
            im_ani = animation.ArtistAnimation(fig1, img_list, interval=50, repeat_delay=3000, blit=True)
            im_ani.save(f'{self.logger.output_dir}/{self.logger.exp_name}_{self.steps_count}.mp4', writer=writer)
            plt.close(fig1)
    
    def logger_checkpoint(self, start_time):
        self.logger.log_tabular('Epoch', self.epoch, step=self.steps_count)
        learner_checkpoint_data = self.learner.checkpoint()
        for key, flag in learner_checkpoint_data.items():
            if flag == 'mm':
                self.logger.log_tabular(key, with_min_and_max=True, step=self.steps_count)
            elif flag == 'avg':
                self.logger.log_tabular(key, average_only=True, step=self.steps_count)
            else:
                print('unknown checkpoint flag')
        # Log info about epoch
        self.logger.log_tabular('TotalEnvInteracts', self.steps_count, step=self.steps_count)
        self.logger.log_tabular('Time', time.time()-start_time, step=self.steps_count)
        self.logger.dump_tabular()

    def wrapup(self,):
        self.logger.wrapup()
        self.buffer.wrapup()

    def run_experiment(self): # main training loop
        self.setup_buffer()
        self.zero_epoch_eval()

        start_time = time.time()
        total_steps = self.config['env_steps_per_epoch'] * self.config['epochs']

        # Main loop: collect experience in env and update/log each epoch
        while self.steps_count < total_steps:
            # Run - produce a single rollout
            ep_len = self.rollout()

            # Train - update the learner
            self.improve(self.config['train_steps_per_rollout'])

            # Evaluate (once per epoch)
            if ((self.steps_count // self.config['env_steps_per_epoch']) > self.epoch):
                # Save statistics of of the buffer
                buff_stats = self.buffer.get_stats()
                if buff_stats: self.logger.cloud_logger.log(buff_stats, self.steps_count)

                self.epoch = self.steps_count // self.config['env_steps_per_epoch']
                # Test the performance of the deterministic version of the agent.
                self.evaluate(n=30)
                self.logger_checkpoint(start_time)
                # Save model every save_epoch_freq epochs
                if (self.epoch % self.config['save_epoch_freq'] == 0) or (self.epoch == self.config['epochs']-1):
                    self.save(self.steps_count)
                    self.logger.upload_assets()
        self.wrapup()

class ImageExperiment(Experiment):
    def setup_buffer(self):
        obs, img = self.sample_obs
        act_dim = self.env.action_space.shape[0]
        obs_dim = obs.shape[0]
        self.buffer = DriveReplayBufferCNN(obs_dim=obs_dim,
                                           act_dim=act_dim,
                                           size=self.config['replay_size'],
                                           img_size=img.shape,
                                           n_cpus=self.config['cpu_workers'])

class OffPolicyImageExperiment(ImageExperiment):
    def setup_buffer(self):
        obs, img = self.sample_obs
        act_dim = self.env.action_space.shape[0]
        obs_dim = obs.shape[0]
        self.buffer = PPOCNNBuffer(obs_dim=obs_dim,
                                   act_dim=act_dim,
                                   size=self.config['env_steps_per_epoch'],
                                   img_size=img.shape,
                                   gamma=self.config['gamma'], 
                                   lam=self.config['lam']
                                  )
    def improve(self):
        self.learner.train()
        metrics = self.learner.learn(self.buffer)
        self.logger.store(**metrics)
        
    def run_experiment(self): # main training loop
        self.setup_buffer()
        self.zero_epoch_eval()

        start_time = time.time()
        total_steps = self.config['env_steps_per_epoch'] * self.config['epochs']

        # Main loop: collect experience in env and update/log each epoch
        while self.steps_count < total_steps:
            # Run - produce a single rollout
            self.rollout()

            # Train - update the learner
            self.improve()

            # Evaluate (once per epoch)
            if ((self.steps_count // self.config['env_steps_per_epoch']) > self.epoch):
                # Save statistics of of the buffer
                buff_stats = self.buffer.get_stats()
                if buff_stats: self.logger.cloud_logger.log(buff_stats, self.steps_count)

                self.epoch = self.steps_count // self.config['env_steps_per_epoch']
                # Test the performance of the deterministic version of the agent.
                self.evaluate(n=30)
                self.logger_checkpoint(start_time)
                # Save model every save_epoch_freq epochs
                if (self.epoch % self.config['save_epoch_freq'] == 0) or (self.epoch == self.config['epochs']-1):
                    self.save(self.steps_count)
                    self.logger.upload_assets()
        self.wrapup()

    def rollout(self):
        exp_count = 0
        while exp_count < self.config['env_steps_per_epoch']:
            o, d = self.env.reset(), False
            ep_ret, ep_len = 0, 0
            while not (d or (ep_len == self.config['max_ep_len'])): # single trajectory rollout
                a, logp_pi, v = self.learner.get_action(o, deterministic=False)
                self.logger.store(VVals=v)

                # Step the env
                o2, r, d, i = self.env.step(a)

                ep_ret += r
                ep_len += 1
                exp_count += 1
                self.steps_count += 1

                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d = False if ep_len==self.config['max_ep_len'] else d
                if i['is_success']: d = True

                # Store experience to replay buffer
                self.buffer.store(o, a, r, v, logp_pi)

                terminal = d or (ep_len == self.config['max_ep_len'])
                if terminal or (exp_count == self.config['env_steps_per_epoch']):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r if d else self.learner.get_action(o2, False)[-1]
                    self.buffer.finish_path(last_val)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    else:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                        break

                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2