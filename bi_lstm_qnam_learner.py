import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from components.episode_buffer import EpisodeBatch
from qnam import QNAMer
from qnam_context import VAE
from torch.optim import RMSprop, Adam


class QNAM_Learner:
    def __init__(self, mac, env, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.env = env
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qnam":
                self.mixer = QNAMer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params_mixer = list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.eval_diff_network = VAE(args.rnn_hidden_dim, args.obs_shape)
        self.params_mixer += list(self.eval_diff_network.parameters())
        self.target_diff_network = VAE(args.rnn_hidden_dim, args.obs_shape)

        if self.args.use_cuda:
            self.eval_diff_network.cuda()
            self.target_diff_network.cuda()

        self.target_diff_network.load_state_dict(
            self.eval_diff_network.state_dict())

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser_mixer = Adam(params=self.params_mixer, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)   # last_actions

        self.mac.init_hidden(batch.batch_size)
        # initial_hidden = self.mac.hidden_states.clone().detach()  #
        # initial_hidden = initial_hidden.reshape(
        #     -1, initial_hidden.shape[-1]).to(self.args.device)  #
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)
        # Qn, hi
        mac_out, hidden, local_qs = self.mac.agent.forward(
            input_here.clone().detach(), self.mac.hidden_states)   # (bs*n, t, h_dim)
        # (bs, t, n, h_dim) mean
        hidden_store = (hidden[0] + hidden[1]) / 2
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[0], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        input_var_here = batch["obs"][:, :-1]  # State
        recon, mean, std = self.eval_diff_network(hidden_store[:, :-1])  # encode + decode --> Mi, zi的分布
        output_vae_here = th.einsum("btnd, btnd->btnd", recon, input_var_here)  # Qi
        recon_loss = F.mse_loss(output_vae_here, input_var_here)  # L_vae
        KL_loss = -0.5 * (1 + th.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()  # D_KL
        entropy_loss = F.l1_loss(recon, target=th.zeros_like(recon), size_average=True)  # L1_norm
        vae_loss = recon_loss + KL_loss + entropy_loss  # L_Gw

        with th.no_grad():
            latent, _, _ = self.eval_diff_network.encode(hidden_store[:, :-1])  # zi

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        # initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        #initial_hidden_target = initial_hidden_target.reshape(
        #    -1, initial_hidden_target.shape[-1]).to(self.args.device)
        target_mac_out, target_hidden, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), self.target_mac.hidden_states)

        target_mac_out = target_mac_out[:, 1:]
        target_hidden_store = (target_hidden[0] + target_hidden[1])/2
        target_hidden_store = target_hidden_store.reshape(
            -1, input_here.shape[1], target_hidden_store.shape[0], target_hidden_store.shape[-1]).permute(0, 2, 1, 3)
        with th.no_grad():
            target_latent, _, _ = self.target_diff_network.encode(target_hidden_store[:, 1:])

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        chosen_action_qvals, coop_alliance, coop_credit = self.mixer(chosen_action_qvals, batch["state"][:, :-1], latent)  # Q_tot = Mixing Network(Qi,S,zi)
        self.env.coop_alliance = coop_alliance
        self.env.coop_credit = coop_credit

        target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_latent)
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals  # y'

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        loss += vae_loss*self.args.beta_vae  # ||Qtot - y'||^2 + β*L_Gw

        print("episode_num: {}, loss:{}".format(episode_num, loss))
        # Optimise
        self.optimiser.zero_grad()
        self.optimiser_mixer.zero_grad()
        loss.backward()
        # 对梯度进行裁剪 防止梯度爆炸
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        grad_norm_mixer = th.nn.utils.clip_grad_norm_(self.params_mixer, self.args.grad_norm_clip)
        self.optimiser.step()
        self.optimiser_mixer.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_diff_network.load_state_dict(
            self.eval_diff_network.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)
