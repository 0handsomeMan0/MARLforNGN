from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import os
from functools import partial


class EpisodeRunner:

    def __init__(self, args, logger, env):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        # self.env.reset()
        self.t = 0

    def run(self, test_mode=False, vae=None, mixer=None):
        self.reset()

        terminated = True
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        #while not terminated:

        pre_transition_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }

        self.batch.update(pre_transition_data, ts=self.t)

        # Pass the entire batch of experiences up till now to the agents
        # Receive the actions for each agent at this timestep in a batch of size 1

        if self.args.evaluate and test_mode and vae:
            actions, hidden, agent_outputs = self.mac.select_actions_vis(self.batch, t_ep=self.t, t_env=self.t_env,
                                                                         test_mode=test_mode)
            mask, _, _ = vae(hidden)
            latent, _, _ = vae.encode(hidden)
            inp_state = self.env.get_state()
            inp_state = th.from_numpy(inp_state).to(self.args.device)
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = agent_outputs.max(dim=2)[0]
            chosen_action_qvals = chosen_action_qvals.unsqueeze(1)
            q_tot, f_i, w, v = mixer(chosen_action_qvals, inp_state, latent, test=True)
            f_i = f_i.squeeze()
            w = w.squeeze()
            v = v.squeeze()
            af_i = f_i * w
            if 'foraging' in self.args.env:
                reward, terminated, env_info = self.env.step_with_vis(actions[0], mask, agent_outputs, af_i, v)
            else:
                reward, terminated, env_info = self.env.step(actions[0])
        elif self.args.evaluate and test_mode:
            actions, hidden, agent_outputs = self.mac.select_actions_vis(self.batch, t_ep=self.t, t_env=self.t_env,
                                                                         test_mode=test_mode)
            inp_state = self.env.get_state()
            inp_state = th.from_numpy(inp_state).to(self.args.device)
            chosen_action_qvals = agent_outputs.max(dim=2)[0]
            chosen_action_qvals = chosen_action_qvals.unsqueeze(1)
            q_tot = mixer(chosen_action_qvals, inp_state).squeeze(0)
            chosen_action_qvals = chosen_action_qvals.squeeze(0)
            reward, terminated, env_info = self.env.step(actions[0])
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            reward, cost, terminated, env_info = self.env.step(actions[0])
            print(env_info)
        episode_return += reward

        post_transition_data = {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }
        self.batch.update(post_transition_data, ts=self.t)
        self.t += 1



        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        return self.batch
