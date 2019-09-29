import os
import uuid
import time
import shutil
import threading
import torch
from multiprocessing import Pool

import numpy as np
import scipy

class Buffer:
    def __init__(self,): pass
    def store(self,): pass
    def sample_batch(self,): pass
    def save(self,): pass
    def load(self,): pass
    def wrapup(self,): pass
    def get_stats(self,): None

class DriveReplayBufferCNN(Buffer):
    """
    A FIFO experience replay buffer for image based DDPG agents.
    The buffer is designed for use on the systems with the limited RAM,
    as it stores the experineces on the drive and loads them based on
    the availability of the CPU workers.
    """

    def __init__(self, obs_dim, act_dim, size, img_size, small_size=5000, n_cpus=4):
        super(DriveReplayBufferCNN, self).__init__()
        self.act_dim = act_dim
        self.img_size = img_size

        self.l_exp_ids = np.zeros(size, dtype=np.int32)
        self.l_acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.l_rews_buf = np.zeros(size, dtype=np.float32)
        self.l_done_buf = np.zeros(size, dtype=np.float32)
        self.l_ptr, self.l_size, self.l_max_size = 0, 0, size

        self.s_acts_buf = np.zeros([small_size, act_dim], dtype=np.float32)
        self.s_rews_buf = np.zeros(small_size, dtype=np.float32)
        self.s_done_buf = np.zeros(small_size, dtype=np.float32)
        self.s1_imgs1_buf = np.zeros([small_size, *img_size], dtype=np.float32)
        self.s1_imgs2_buf = np.zeros([small_size, *img_size], dtype=np.float32)
        self.s2_imgs1_buf = np.zeros([small_size, *img_size], dtype=np.float32)
        self.s2_imgs2_buf = np.zeros([small_size, *img_size], dtype=np.float32)
        self.s_ptr, self.s_size, self.s_max_size = 0, 0, small_size

        folder_id = str(uuid.uuid4())[:8]
        self.storage_folder = f'tmp_buffer{folder_id}/'
        if os.path.exists(self.storage_folder):
            shutil.rmtree(self.storage_folder)
        os.makedirs(self.storage_folder)

        self.s_buffer_is_full = False
        self.l_buffer_is_full = False
        self.exp_queue = []
        self.latest_stored_experience_id = 0
        self.resamples_count = 0
        self.replace_time = True
        self.num_proc = n_cpus
        self.pool = Pool(processes=self.num_proc)
        self.pool_res = None
        self.kill_thread = False

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True # Daemonize thread
        thread.start() # Start the execution

    def run(self):
        while not self.kill_thread:
            if len(self.exp_queue)>self.num_proc:
                self.save_to_drive()
            if self.replace_time and self.s_buffer_is_full:
                self.load_sample_from_drive()
            time.sleep(0.1)

    def save_exp(self, data):
        (s1_img1, s1_img2), (s2_img1, s2_img2), uid = data
        torch.save(s1_img1, self.storage_folder + f'{uid}_s1_img1.pt')
        torch.save(s1_img2, self.storage_folder + f'{uid}_s1_img2.pt')
        torch.save(s2_img1, self.storage_folder + f'{uid}_s2_img1.pt')
        torch.save(s2_img2, self.storage_folder + f'{uid}_s2_img2.pt')

    def load_exp(self, idx):
        s1_img1 = torch.load(self.storage_folder + f'{idx}_s1_img1.pt')
        s1_img2 = torch.load(self.storage_folder + f'{idx}_s1_img2.pt')
        s2_img1 = torch.load(self.storage_folder + f'{idx}_s2_img1.pt')
        s2_img2 = torch.load(self.storage_folder + f'{idx}_s2_img2.pt')
        return idx, (s1_img1, s1_img2), (s2_img1, s2_img2)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict.clear()
        self_dict['storage_folder'] = self.storage_folder
        return self_dict

    def save_to_drive(self):
        if self.pool_res is not None: self.pool_res.get()
        exps, self.exp_queue = self.exp_queue[:self.num_proc], self.exp_queue[self.num_proc:]
        imgs  = [(s, s_next, uid) for (s, a, r, s_next, d, uid) in exps]
        exp_states = [(a, r, d, uid) for (s, a, r, s_next, d, uid) in exps]
        self.pool_res = self.pool.map_async(self.save_exp, imgs)

        act, rew, done, uid = zip(*exp_states)

        n_exps = len(exp_states)
        store_idxs = np.arange(self.l_ptr,self.l_ptr+n_exps)%self.l_max_size
        if self.l_buffer_is_full:
            old_files = self.l_exp_ids[store_idxs]
            for f_uid in old_files:
                os.remove(self.storage_folder + f'{f_uid}.pt')
                os.remove(self.storage_folder + f'{f_uid}_next.pt')
        if not self.l_buffer_is_full and self.l_ptr+n_exps >= self.l_max_size:
            print('Large buffer is filled up, starting removing old experiencs')
            self.l_buffer_is_full = True
        self.l_exp_ids[store_idxs] = uid
        self.l_acts_buf[store_idxs] = act
        self.l_rews_buf[store_idxs] = rew
        self.l_done_buf[store_idxs] = done
        self.l_ptr = (self.l_ptr+n_exps) % self.l_max_size
        self.l_size = min(self.l_size+n_exps, self.l_max_size)

    def load_sample_from_drive(self):
        if self.pool_res is not None: self.pool_res.get()
        all_idxs = np.arange(self.l_exp_ids[:self.l_size].shape[0])
        pos_idxs = np.random.choice(all_idxs, size=self.num_proc, replace=False)
        idxs = self.l_exp_ids[pos_idxs]
        idx_map = {ix:pos_ix for ix, pos_ix in zip(idxs, pos_idxs)}
        self.pool_res = self.pool.map(self.load_exp, idxs)

        for im_obs in self.pool_res:
            idx, (s1_img1, s1_img2), (s2_img1, s2_img2) = im_obs
            idx = idx_map[idx]
            obs, next_obs = (s1_img1, s1_img2), (s2_img1, s2_img2)
            act, rew, done = self.l_acts_buf[idx], self.l_rews_buf[idx], self.l_done_buf[idx]
            self.store_locally(obs, act, rew, next_obs, done)
            self.resamples_count += 1
        self.pool_res = None

    def store_locally(self, obs, act, rew, next_obs, done):
        s1_img1, s1_img2 = obs
        s2_img1, s2_img2 = next_obs
        self.s_acts_buf[self.s_ptr] = act
        self.s_rews_buf[self.s_ptr] = rew
        self.s_done_buf[self.s_ptr] = done
        self.s1_imgs1_buf[self.s_ptr] = s1_img1
        self.s1_imgs2_buf[self.s_ptr] = s1_img2
        self.s2_imgs1_buf[self.s_ptr] = s2_img1
        self.s2_imgs2_buf[self.s_ptr] = s2_img2
        self.s_ptr = (self.s_ptr+1) % self.s_max_size
        self.s_size = min(self.s_size+1, self.s_max_size)

    def store(self, obs, act, rew, next_obs, done):
        if not self.s_buffer_is_full:
            self.store_locally(obs, act, rew, next_obs, done)
            if self.s_ptr+1 == self.s_max_size:
                print('Small buffer is filled up, starting sampling from the drive')
                self.s_buffer_is_full = True
        exp_tuple = (obs, act, rew, next_obs, done, self.latest_stored_experience_id)
        self.exp_queue.append(exp_tuple)
        self.latest_stored_experience_id += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.s_size, size=batch_size)
        return dict(obs1=(self.s1_imgs1_buf[idxs],self.s1_imgs2_buf[idxs]),
                    obs2=(self.s2_imgs1_buf[idxs],self.s2_imgs2_buf[idxs]),
                    acts=self.s_acts_buf[idxs],
                    rews=self.s_rews_buf[idxs],
                    done=self.s_done_buf[idxs])

    def get_stats(self,):
        stats = dict()
        stats["exp_queue_size"] = len(self.exp_queue)
        stats["latest_stored_experience_id"] = self.latest_stored_experience_id
        stats["resamples_count"] = self.resamples_count
        return stats

    def wrapup(self):
        print(self.get_stats())
        self.kill_thread = True
        time.sleep(1.0)
        # clean up the drive
        if os.path.exists(self.storage_folder):
            shutil.rmtree(self.storage_folder)
        print('ReplayBuffer stopped cleanly.')