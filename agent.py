from lightsaber.rl.trainer import AgentInterface
import network
import build_graph
import lightsaber.tensorflow.util as util
import numpy as np
import tensorflow as tf


class Agent(AgentInterface):
    def __init__(self, actor, critic, obs_dim,
                num_actions, replay_buffer, batch_size=16, gamma=0.9):
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.last_obs = None
        self.t = 0
        self.exploration = 3
        self.replay_buffer = replay_buffer

        self._act,\
        self._train_actor,\
        self._train_critic,\
        self._update_actor_target,\
        self._update_critic_target = build_graph.build_train(
            actor=actor,
            critic=critic,
            obs_dim=obs_dim,
            num_actions=num_actions,
            gamma=gamma
        )

    def act(self, obs, reward, training):
        obs = obs[0]
        action = np.random.normal(self._act([obs])[0], self.exploration)
        action = np.clip(action, -2, 2)

        if training and self.t > 10000:
            self.exploration *= 0.9995
            obs_t,\
            actions,\
            rewards,\
            obs_tp1,\
            dones = self.replay_buffer.sample(self.batch_size)
            actor_error = self._train_actor(obs_t)
            critic_error = self._train_critic(
                obs_t,
                actions,
                rewards,
                obs_tp1,
                dones
            )
            self._update_actor_target()
            self._update_critic_target()

        if training and self.last_obs is not None:
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=False
            )

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        return action

    def stop_episode(self, obs, reward, done=False, training=True):
        obs = obs[0]
        if training:
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=done
            )
        self.last_obs = None
        self.last_action = []
