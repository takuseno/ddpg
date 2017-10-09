import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(actor, critic, obs_dim, num_actions, bound, optimizer,
                grad_norm_clipping=10.0, gamma=1.0, scope='ddpg', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_t')
        act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        # actor network
        policy_t = actor(obs_t_input, num_actions, bound, scope='actor')
        actor_func_vars = util.scope_vars(util.absolute_scope_name('actor'), trainable_only=True)

        # target actor network
        policy_tp1 = actor(obs_tp1_input, num_actions, bound, scope='target_actor')
        target_actor_func_vars = util.scope_vars(util.absolute_scope_name('target_actor'), trainable_only=True)

        # critic network
        q_t = critic(obs_t_input, act_t_ph, num_actions, scope='critic')
        q_t_with_actor = critic(obs_t_input, policy_t, num_actions, scope='critic', reuse=True)
        critic_func_vars = util.scope_vars(util.absolute_scope_name('critic'), trainable_only=True)

        # target critic network
        q_tp1 = critic(obs_tp1_input, policy_tp1, num_actions, scope='target_critic')
        target_critic_func_vars = util.scope_vars(util.absolute_scope_name('target_critic'), trainable_only=True)

        # loss
        with tf.variable_scope('target_q'):
            target_q = rew_t_ph + (1 - done_mask_ph) * gamma * q_tp1
        critic_loss = tf.reduce_mean(tf.square(target_q - q_t), name='critic_loss')
        actor_loss = tf.negative(tf.reduce_mean(q_t_with_actor), name='actor_loss')

        # optimize operations
        critic_optimizer = tf.train.AdamOptimizer(0.001)
        critic_gradients = critic_optimizer.compute_gradients(critic_loss, var_list=critic_func_vars)
        critic_optimize_expr = critic_optimizer.apply_gradients(critic_gradients)
        actor_optimizer = tf.train.AdamOptimizer(0.001)
        actor_gradients = actor_optimizer.compute_gradients(actor_loss, var_list=actor_func_vars)
        actor_optimize_expr = actor_optimizer.apply_gradients(actor_gradients)

        # update target operations
        update_target_expr = []
        # assign critic variables to target critic variables
        for var, var_target in zip(sorted(critic_func_vars, key=lambda v: v.name),
                                    sorted(target_critic_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        # assign actor variables to target actor variables
        for var, var_target in zip(sorted(actor_func_vars, key=lambda v: v.name),
                                    sorted(target_actor_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # action theano-style function
        act = util.function(inputs=[obs_t_input], outputs=policy_t)

        # train theano-style function
        train_actor = util.function(
            inputs=[
                obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph
            ],
            outputs=[actor_loss],
            updates=[actor_optimize_expr]
        )
        train_critic = util.function(
            inputs=[
                obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph
            ],
            outputs=[critic_loss],
            updates=[critic_optimize_expr]
        )

        # update target theano-style function
        update_target = util.function([], [], updates=[update_target_expr])

        return act, train_actor, train_critic, update_target
