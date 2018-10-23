import argparse
import gym
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn

# import tensorboardX
# from utils import slugify
from envs import create_atari_env
from a3c_model import ActorCritic
from a3c_optim import SharedAdam
from a3c_train import train
from a3c_test import test
# from reward_model import OriginalEnvironmentReward, EpisodeLogger

# from a3c import NNPolicy, SharedAdam
# from train import train_a3c

def get_args():
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('--env_name', default='Breakout-v4', type=str, help='gym environment')
	parser.add_argument('--workers', default=4, type=int, help='number of workers to train with')
	parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
	parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
	parser.add_argument('--num-steps', default=20, type=int, help='')
	parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
	parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
	parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
	parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
	parser.add_argument('--max-grad-norm', type=float, default=50, help='')
	parser.add_argument('--entropy-coef', default=0.01, type=float, help='entropy term coefficient (default: 0.01)')
	parser.add_argument('--value-loss-coef', default=0.5, type=float, help='')
	parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
	parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
	parser.add_argument('--save_dir', default='/tmp/test_a3c', type=str, help='saved directory')
	parser.add_argument('--max-episode-length', default=8e7, type=int)
	parser.add_argument('--no_shared', default=False, type=bool, help='')
	return parser.parse_args()


# def printlog(args, s, end='\n', mode='a'):
#     print(s, end=end)
#     f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

args = get_args()
# print('args', args)
# env = make_env(args.env)
env = create_atari_env(args.env_name)
shared_model = ActorCritic(
	env.observation_space.shape[0], env.action_space)
shared_model.share_memory()

if args.no_shared:
	optimizer = None
else:
	optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
	optimizer.share_memory()
# exp_name = slugify(args.env)
# print('num_actions', env.action_space.n)
# n_pretrain_labels = 0

# episode_logger = EpisodeLogger('test')

# reward_model = OriginalEnvironmentReward(episode_logger)
# args.pretrain_iters = 0  # Don't bother pre-training a traditional RL agent

# reward_model.try_to_load_model_from_checkpoint()

# reward_model.train(args.pretrain_iters, report_frequency=25)
# reward_model.save_model_checkpoint()

# num_actions = gym.make(args.env).action_space.n

# info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
# info['frames'] += shared_model.try_load(args.save_dir) * 1e6
# if int(info['frames'].item()) == 0: 
# 	printlog(args,'', end='', mode='w') # clear log file

# train_a3c(args, reward_model)
processes = []

counter = mp.Value('i', 0)
lock = mp.Lock()

p = mp.Process(target=test, args=(args.workers, args, shared_model, counter))
p.start()
processes.append(p)

for rank in range(0, args.workers):
    p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
    p.start(); 
    processes.append(p)
for p in processes: 
	p.join()