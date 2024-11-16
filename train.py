from environment import Environment
from components.transforms import OneHot
import torch as th
from components.episode_buffer import ReplayBuffer
from basic_controller import BasicMAC
from arguments import Arguments
from bi_lstm_qnam_learner import QNAM_Learner
from utils.logging import get_logger
import time
from episode_runner import EpisodeRunner



def train():
    batch_size = 1
    max_request_num = 390  # 每轮训练最大请求数据量 1% 5% 10% 15% 20% 25% 30%
    request_dim = 3
    cache_size = 390  # 缓存大小 3900 5% 10% 15% 20% 25% 30%
    n_agents = 10  # sbs数量
    n_actions = 2  # 动作空间[0 缓存, 1 不缓存]
    state_shape = n_agents * (cache_size + max_request_num * request_dim)  # 环境内部状态(所有sbs的缓存状态 + 当前请求的数据)
    obs_shape = cache_size   # 可观测的环境(自己的缓存信息)
    buffer_size = 10  # 经验池大小
    max_seq_length = 1  # 最大序列长度
    burn_in_period = 1  # 用于确定在开始从回放缓冲区中采样经验之前，需要有多少经验被添加到缓冲区中
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    args = Arguments(batch_size=batch_size,
                     buffer_size=buffer_size,
                     device=device,
                     n_actions=n_actions,
                     n_agents=n_agents,
                     obs_shape=obs_shape,
                     state_shape=state_shape)

    logger = get_logger()

    # 1.初始化网络环境
    environment = Environment(sbs_number=n_agents, cache_size=cache_size, n_actions=n_actions, n_agents=n_agents, sigma=0.5,
                              episode_limit=max_seq_length, max_request_num=max_request_num, request_dim=request_dim, gamma=0.2)

    runner = EpisodeRunner(args=args, logger=logger, env=environment)

    # 2.初始化ReplayBuffer、BasicMAC、QNAM_Learner
    scheme = {
        "state": {"vshape": state_shape},
        "obs": {"vshape": obs_shape, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, buffer_size, max_seq_length + 1,
                          burn_in_period=burn_in_period,
                          preprocess=preprocess,
                          device=device)

    mac = BasicMAC(buffer.scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    learner = QNAM_Learner(mac, environment, logger, args)
    if args.use_cuda:
        learner.cuda()

    # 3.运行并训练
    episode = 0

    start_time = time.time()
    last_time = start_time

    logger.info("Beginning training for {} timesteps".format(args.t_max))


    while episode <= args.t_max:
        episode_batch = runner.run(test_mode=False)  # 运行一个step，收集数据
        buffer.insert_episode_batch(episode_batch)  # 插入buffer

        if buffer.can_sample():
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, episode)
            episode = episode + 1


    logger.info("Finished Training")


if __name__ == '__main__':
    train()