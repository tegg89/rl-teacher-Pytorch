import glob
import multiprocessing as mp

from time import sleep

import numpy as np
import torch
import torch.nn as nn


class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()

        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class Actor(mp.Process):
    def __init__(self, args, task_q, result_q, make_env, stacked_frames, seed):
        mp.Process.__init__(self)
        # print("entered Actor")
        self.args = args
        self.make_env = make_env
        self.stacked_frames = stacked_frames
        self.seed = seed
        self.task_q = task_q
        self.result_q = result_q
        self.max_timesteps_per_episode = args.max_timesteps_per_episode

        
        # self.run()
    # def set_policy(self, weights):

    def run(self):
        # print('run')
        self.env = self.make_env(self.args.env)
        self.env.seed = self.seed
        # print('Created environment')

        self.obs_size = list(self.env.observation_space.shape)
        # print('self.obs_size', self.obs_size)
        # if stacked_frames > 0:
        #   self.obs_size += [stacked_frames]
        self.hidden_size = self.args.hidden
        self.num_actions = self.env.action_space.n
        # print('self.num_actions', self.num_actions)

        # self.env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)



        # self.model = NNPolicy(channels=1, memsize=256, num_actions=self.num_actions)
        # self.shared_model = self.model.share_memory()
        # self.shared_optimizer = SharedAdam(self.shared_model.parameters(), lr=self.args.lr)

        # for param, shared_param in zip(self.model.parameters(), self.shared_model.parameters()):
        #     if shared_param.grad is None: 
        #         shared_param._grad = param.grad # sync gradients with shared model
        # self.shared_optimizer.step()
        # print('stepping')

        # while True:
        #     print('enter while')
        #     next_task = self.task_q.get(block=True)
        #     print('next_task')
        #     if next_task == "do_rollout":
        #         print("do_rollout")
        #         path = self.rollout()
        #         self.task_q.task_done()
        #         self.result_q.put(path)
        #     elif next_task == "kill":
        #         print("kill")
        #         self.task_q.task_done()
        #         break
        #     else:
        #         print("self.set_policy(next_task)")

        #         sleep(0.1)
        #         self.task_q.task_done()

        def rollout(self):
            print("Actor.rollout")
            obs, human_obs, actions, rewards = [], [], [], []
            values, logps = [], [] # save values for computing gradients

            for i in range(self.max_timesteps_per_episode):

                state = torch.tensor(prepro(self.env.reset())) # get first state
                hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
                obs.append(state) # saved for first state (rnn)

                for step in range(self.args.rnn_steps):
                    episode_length += 1
                    value, logit, hx = model((state.view(1,1,80,80), hx))
                    logp = F.log_softmax(logit, dim=-1)

                    action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
                    state, reward, done, info = env.step(action.numpy()[0])
                    print("info", info)

                    if self.args.render: 
                        env.render()

                    state = torch.tensor(prepro(state))

                    epr += reward
                    reward = np.clip(reward, -1, 1) # reward
                    done = done or episode_length >= 1e4 # don't playing one ep for too long

                    obs.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    human_obs.append(info.get("human_obs"))
                    values.append(value)
                    logps.append(logp)

                    # info['frames'].add_(1)
                    # num_frames = int(info['frames'].item())
                    # if num_frames % 2e6 == 0: # save every 2M frames
                    #     printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                    #     torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

                    if done: # update shared data
                        info['episodes'] += 1
                        interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                        info['run_epr'].mul_(1-interp).add_(interp * epr)
                        info['run_loss'].mul_(1-interp).add_(interp * eploss)

                        state = torch.tensor(prepro(env.reset()))

                    # if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                    #     elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    #     printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    #         .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    #         info['run_epr'].item(), info['run_loss'].item()))
                    #     last_disp_time = time.time()

                    # if done: # maybe print info.
                        # episode_length, epr, eploss = 0, 0, 0
                        # state = torch.tensor(prepro(env.reset()))
                        

                    
                        path = {
                            "obs": np.array(obs),
                            "actions": np.array(actions),
                            "rewards": np.array(rewards),
                            "human_obs": np.array(human_obs),}

                        return path

        rollout()



class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)



class ParallelRollout(object):
    def __init__(self, args, make_env, stacked_frames, reward_predictor, max_timesteps_per_episode, seed):
        self.num_workers = args.workers
        self.predictor = reward_predictor

        self.tasks_q = mp.JoinableQueue()
        self.results_q = mp.Queue()

        self.actors = []

        for _ in range(args.workers):
            # print('rank')
            a = mp.Process(
                target=Actor, 
                args=(args, self.tasks_q, self.results_q, make_env, stacked_frames, seed))
            self.actors.append(a)
            # print('actors ready')
            # print(self.actors)
            a.start()
        for a in self.actors: 
            a.join()

        # for i in range(self.num_workers):
        #     new_seed = seed * 1000 + i  # Give each actor a uniquely seeded env
        #     self.actors.append(Actor(
        #         args, self.tasks_q, self.results_q, make_env, stacked_frames, seed))
        #     # print('self.actors', self.actors)

        # for a in self.actors:
        #     # print('a')
        #     a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first iteration  TODO OLD
        self.average_timesteps_in_episode = 1000

    def rollout(self, timesteps):
        print('rollout - {}'.format(timesteps))
        start_time = time()
        # keep 20,000 timesteps per update  TODO OLD
        # TODO Run by number of rollouts rather than time
        num_rollouts = int(timesteps / self.average_timesteps_in_episode)
        print('ParallelRollout/num_rollouts: {}'.format(num_rollouts))

        for _ in range(num_rollouts):
            self.tasks_q.put("do_rollout")
        self.tasks_q.join()

        paths = []
        for _ in range(num_rollouts):
            path = self.results_q.get()

            ################################
            #  START REWARD MODIFICATIONS  #
            ################################
            path["original_rewards"] = path["rewards"]
            path["rewards"] = self.predictor.predict_reward(path)
            self.predictor.path_callback(path)
            ################################
            #   END REWARD MODIFICATIONS   #
            ################################

            paths.append(path)

        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)

        return paths, time() - start_time

    def set_policy_weights(self, parameters):
        for i in range(self.num_workers):
            self.tasks_q.put(parameters)
        self.tasks_q.join()

    def end(self):
        for i in range(self.num_workers):
            self.tasks_q.put("kill")
