
class Arguments:
    def __init__(self,
                 action_selector='epsilon_greedy',
                 agent='rnn_sd',
                 agent_output_type='q',
                 alg_name='qnam',
                 ally_num=0,
                 batch_size=32,
                 batch_size_run=1,
                 beta_vae=0.1,
                 buffer_cpu_only=True,
                 buffer_size=5000,
                 burn_in_period=100,
                 checkpoint_path='',
                 critic_lr=0.0005,
                 depth=2,
                 device='cpu',
                 double_q=True,
                 emb=32,
                 enemy_num=10,
                 env='stag_hunt',
                 epsilon_anneal_time=50000,
                 epsilon_finish=0.05,
                 epsilon_start=1.0,
                 evaluate=False,
                 gamma=0.99,
                 gated=False,
                 gpu_id='0',
                 grad_norm_clip=10,
                 heads=3,
                 hypernet_embed=64,
                 hypernet_layers=2,
                 label='default_label',
                 learner='qnam_learner',
                 learner_log_interval=10000,
                 load_step=0,
                 local_results_path='results',
                 log_interval=10000,
                 lr=0.0005,
                 mac='basic_mac',
                 map_name='lbf-4-2',
                 mix_nary=[1, 2],
                 mixer='qnam',
                 mixing_embed_dim=32,
                 n_actions=2,
                 n_agents=10,
                 name='qnam',
                 obs_agent_id=False,
                 obs_last_action=True,
                 obs_shape=100,
                 optim_alpha=0.99,
                 optim_eps=1e-05,
                 repeat_id=1,
                 rnn_hidden_dim=64,
                 runner='episode',
                 runner_log_interval=10000,
                 save_model=False,
                 save_model_interval=2000000,
                 save_replay=False,
                 seed=1,
                 state_shape=1000,
                 t_max=1010000,
                 target_update_interval=100,
                 test_greedy=True,
                 test_interval=10000,
                 test_nepisode=16,
                 token_dim=5,
                 training_iters=1,
                 unique_token='qnam__2024-05-30_10-34-19',
                 unit_dim=100,
                 use_cuda=False,
                 use_tensorboard=False
                 ):

        self.action_selector = action_selector
        self.agent = agent
        self.agent_output_type = agent_output_type
        self.alg_name = alg_name
        self.ally_num = ally_num
        self.batch_size = batch_size
        self.batch_size_run = batch_size_run
        self.beta_vae = beta_vae
        self.buffer_cpu_only = buffer_cpu_only
        self.buffer_size = buffer_size
        self.burn_in_period = burn_in_period
        self.checkpoint_path = checkpoint_path
        self.critic_lr = critic_lr
        self.depth = depth
        self.device = device
        self.double_q = double_q
        self.emb = emb
        self.enemy_num = enemy_num
        self.env = env
        self.epsilon_anneal_time = epsilon_anneal_time
        self.epsilon_finish = epsilon_finish
        self.epsilon_start = epsilon_start
        self.evaluate = evaluate
        self.gamma = gamma
        self.gated = gated
        self.gpu_id = gpu_id
        self.grad_norm_clip = grad_norm_clip
        self.heads = heads
        self.hypernet_embed = hypernet_embed
        self.hypernet_layers = hypernet_layers
        self.label = label
        self.learner = learner
        self.learner_log_interval = learner_log_interval
        self.load_step = load_step
        self.local_results_path = local_results_path
        self.log_interval = log_interval
        self.lr = lr
        self.mac = mac
        self.map_name = map_name
        self.mix_nary = mix_nary
        self.mixer = mixer
        self.mixing_embed_dim = mixing_embed_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.name = name
        self.obs_agent_id = obs_agent_id
        self.obs_last_action = obs_last_action
        self.obs_shape = obs_shape
        self.optim_alpha = optim_alpha
        self.optim_eps = optim_eps
        self.repeat_id = repeat_id
        self.rnn_hidden_dim = rnn_hidden_dim
        self.runner = runner
        self.runner_log_interval = runner_log_interval
        self.save_model = save_model
        self.save_model_interval = save_model_interval
        self.save_replay = save_replay
        self.seed = seed
        self.state_shape = state_shape
        self.t_max = t_max
        self.target_update_interval = target_update_interval
        self.test_greedy = test_greedy
        self.test_interval = test_interval
        self.test_nepisode = test_nepisode
        self.token_dim = token_dim
        self.training_iters = training_iters
        self.unique_token = unique_token
        self.unit_dim = unit_dim
        self.use_cuda = use_cuda
        self.use_tensorboard = use_tensorboard