import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf


class Agent(object):
    def __init__(self, actor, critic, obs_dim, num_actions, replay_buffer, bound,
            lr=2.5e-4, batch_size=32, gamma=0.99, target_network_update_freq=20):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_network_update_freq = target_network_update_freq
        self.last_obs = None
        self.t = 0
        self.replay_buffer = replay_buffer

        act, train, update_target = build_graph.build_train(
            actor=actor,
            critic=critic,
            obs_dim=obs_dim,
            num_actions=num_actions,
            bound=bound,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10.0
        )
        self._act = act
        self._train = train
        self._update_target = update_target

    def act(self, obs):
        normalized_obs = np.zeros((1, 84, 84, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        action = self._act(normalized_obs)[0]
        return action

    def act_and_train(self, obs, reward):
        action = self._act([obs])[0]

        if self.t > self.batch_size:
            obs_t, actions, rewards, obs_tp1, dones = self.replay_buffer.sample(self.batch_size)
            obs_t = np.array(obs_t, dtype=np.float32) / 255.0
            obs_tp1 = np.array(obs_tp1, dtype=np.float32) / 255.0
            errors = self._train(obs_t, actions, rewards, obs_tp1, dones)
            self._update_target()

        if self.last_obs is not None:
            self.replay_buffer.append(obs_t=self.last_obs,
                    action=self.last_action, reward=reward, obs_tp1=obs, done=False)

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        self.replay_buffer.append(obs_t=self.last_obs,
                action=self.last_action, reward=reward, obs_tp1=obs, done=done)
        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_action = []
