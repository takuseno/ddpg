import argparse
import cv2
import gym
import copy
import os
import numpy as np
import tensorflow as tf

from lightsaber.tensorflow.util import initialize
from lightsaber.tensorflow.log import TfBoardLogger
from lightsaber.rl.replay_buffer import ReplayBuffer
from lightsaber.rl.trainer import Trainer
from lightsaber.rl.env_wrapper import EnvWrapper
from network import make_actor_network, make_critic_network
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--log', type=str, default=date)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    logdir = os.path.join(os.path.dirname(__file__), 'logs/{}'.format(args.log))

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    env = EnvWrapper(env, r_preprocess=lambda r: r / 10.0)

    actor = make_actor_network([64, 64])
    critic = make_critic_network()
    replay_buffer = ReplayBuffer(10 ** 5)

    sess = tf.Session()
    sess.__enter__()

    agent = Agent(actor, critic, obs_dim, n_actions, replay_buffer)

    initialize()

    saver = tf.train.Saver()
    if args.load is not None:
        saver.restore(sess, args.load)

    train_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(train_writer)
    logger.register('reward', dtype=tf.int32)
    end_episode = lambda r, t, e: logger.plot('reward', r, t)

    trainer = Trainer(
        env=env,
        agent=agent,
        render=args.render,
        end_episode=end_episode,
        training=not args.demo
    )

    trainer.start()

if __name__ == '__main__':
    main()
