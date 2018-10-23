# [Deep reinforcement learning with human preferences](https://arxiv.org/abs/1706.03741) (WIP)

This paper describes that achieving better performance of agents by asking human judgment about more optimal behavior.
Mujoco agents, for instance, requires agents to move to a horizontal direction which gives higher rewards.
Then, although each agent performs differently and weirdly, it can get high rewards which are not an optimistic behavior in the real world.
While the middle of performing a learning behavior, the program asks human's preferences to choose which one has better performance.
This mechanism is similar to the "Ideal World Cup".

This repository is trying to reproduce the corresponding paper. Currently, it is under construction and will keep maintaining.
For the convenience, Atari environment is used instead of Mujoco. 
I have referred to the original repository which is shown below:

* [https://github.com/ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)

* [https://github.com/machine-intelligence/rl-teacher-atari](https://github.com/machine-intelligence/rl-teacher-atari)
