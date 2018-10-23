import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from envs import create_atari_env
from a3c_model import ActorCritic
		

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def create_path(obs, human_obs, action_dists, rewards, actions):
	path = {
		"obs": np.array(obs),
		"human_obs": np.array(human_obs),
		"action_dist": np.array(action_dists),
		# "logstd_action_dist": np.concatenate(log_prob.data.numpy()),
		"rewards": np.array(rewards),
		"actions": np.array(actions),}

	# import _pickle as pickle

	# with open("./path.txt", "wb") as f:
	# 	f.write(pickle.dumps(path))

	# print(path["obs"], path["action_dist"])

	return path

def train(rank, args, shared_model, counter, lock, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    obs = []

    state = env.reset()
    state = torch.from_numpy(state)
    obs.append(obs)
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        
        human_obs = []
        action_dists = []
        values = []
        log_probs = []
        actions = []
        rewards = []
        entropies = []

        '''Requiring trajectories:
		value, logit, (hx, cx)
        '''

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0)),
                                            (hx, cx)))
            # print('value', value.data.numpy())
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            # print('log_prob', log_prob.data.numpy())
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            # action_dists.append(prob.data.numpy())
            action_dists.append(prob)
            # print('action_dists', action_dists)

            action = prob.multinomial().data
            actions.append(action)
            # print('actions', np.array(actions))
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, info = env.step(action.numpy())
            

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            
            state = torch.from_numpy(state)
            obs.append(state)
            # values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            human_obs.append(info.get("human_obs"))

            # print('log_probs', np.concatenate(log_probs))
            # print('human_obs', human_obs)
            # print('action_dists', action_dists)
            if done:
            	# path = create_path(obs, human_obs, action_dists, rewards, actions)
            	# print('create_path',)
            	break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        path = create_path(obs, human_obs, action_dists, rewards, actions)
        print('create_path')
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
