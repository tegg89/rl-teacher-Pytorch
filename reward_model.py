from copy import deepcopy
import os.path as osp
from collections import deque

import numpy as np
from tensorboardX import SummaryWriter

class EpisodeLogger():
    """Tracks and logs agent performance"""

    def __init__(self, name, timesteps_per_summary=int(1e3)):
        logs_path = osp.expanduser('~/tb/rl-teacher-pytorch/%s' % (name))
        self.writer = SummaryWriter(logs_path)

        self.summary_count = 0
        self.timesteps_per_summary = timesteps_per_summary

        self._timesteps_elapsed = 0
        self._timesteps_since_last_summary = 0

        self.last_n_scores = deque(maxlen=100)
        

    @property
    def timesteps_elapsed(self):
        return self._timesteps_elapsed

    def log_episode(self, path):
        path_length = len(path["obs"])
        self._timesteps_elapsed += path_length
        self._timesteps_since_last_summary += path_length

        if 'new' in path:  # PPO puts multiple episodes into one path
            path_count = np.sum(path["new"])
            for _ in range(path_count):
                self.last_n_scores.append(np.sum(path["original_rewards"]).astype(float) / path_count)
        else:
            self.last_n_scores.append(np.sum(path["original_rewards"]).astype(float))

        if self._timesteps_since_last_summary >= self.timesteps_per_summary:
            self.summary_count += 1
            self.log_simple("agent/true_reward_per_episode", np.mean(self.last_n_scores))
            self.log_simple("agent/total_steps", self._timesteps_elapsed)
            self._timesteps_since_last_summary -= self.timesteps_per_summary
            self.flush()

    def log_simple(self, tag, simple_value, debug=False):
        self.writer.add_scalar(tag=tag, scalar_value=simple_value, global_step=self.summary_count)
        if debug:
            print("%s    =>    %s" % (tag, simple_value))


class RewardModel(object):
    def __init__(self, episode_logger):
        self._episode_logger = episode_logger

    def predict_reward(self, path):
        raise NotImplementedError()  # Must be overridden

    def path_callback(self, path):
        self._episode_logger.log_episode(path)

    def train(self, iterations=1, report_frequency=None):
        pass  # Doesn't require training by default

    def save_model_checkpoint(self):
        pass  # Nothing to save

    def try_to_load_model_from_checkpoint(self):
        pass  # Nothing to load

class OriginalEnvironmentReward(RewardModel):
    """Model that always gives the reward provided by the environment."""

    def predict_reward(self, path):
        return path["original_rewards"]