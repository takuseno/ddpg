import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(actor, critic, obs_dim,
        num_actions, gamma=1.0, scope='ddpg', tau=0.001, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # input placeholders
        obs_t_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_t')
        act_t_ph = tf.placeholder(tf.float32, [None, num_actions], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_input = tf.placeholder(tf.float32, [None, obs_dim], name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        # actor network
        policy_t = actor(obs_t_input, num_actions, scope='actor')
        actor_func_vars = util.scope_vars(
            util.absolute_scope_name('actor'),
            trainable_only=True
        )

        # target actor network
        policy_tp1 = actor(obs_tp1_input, num_actions, scope='target_actor')
        target_actor_func_vars = util.scope_vars(
            util.absolute_scope_name('target_actor'),
            trainable_only=True
        )

        # critic network
        q_t = critic(obs_t_input, act_t_ph, num_actions, scope='critic')
        q_t_with_actor = critic(
            obs_t_input,
            policy_t,
            num_actions,
            scope='critic',
            reuse=True
        )
        critic_func_vars = util.scope_vars(
            util.absolute_scope_name('critic'),
            trainable_only=True
        )

        # target critic network
        q_tp1 = critic(
            obs_tp1_input,
            policy_tp1,
            num_actions,
            scope='target_critic'
        )
        target_critic_func_vars = util.scope_vars(
            util.absolute_scope_name('target_critic'),
            trainable_only=True
        )

        # loss
        with tf.variable_scope('target_q'):
            v = (1 - done_mask_ph) * gamma * tf.stop_gradient(q_tp1)
            target_q = rew_t_ph + v
        critic_loss = tf.reduce_mean(
            tf.square(target_q - q_t),
            name='critic_loss'
        )
        actor_loss = -tf.reduce_mean(
            q_t_with_actor,
            name='actor_loss'
        )

        # optimize operations
        critic_optimizer = tf.train.AdamOptimizer(0.001)
        critic_optimize_expr = critic_optimizer.minimize(
            critic_loss,
            var_list=critic_func_vars
        )
        actor_optimizer = tf.train.AdamOptimizer(0.0001)
        actor_optimize_expr = actor_optimizer.minimize(
            actor_loss,
            var_list=actor_func_vars
        )

        # update critic target operations
        with tf.variable_scope('update_critic_target'):
            update_critic_target_expr = []
            sorted_vars = sorted(
                critic_func_vars,
                key=lambda v: v.name
            )
            sorted_target_vars = sorted(
                target_critic_func_vars,
                key=lambda v: v.name
            )
            # assign critic variables to target critic variables
            for var, var_target in zip(sorted_vars, sorted_target_vars):
                new_var = tau * var + (1 - tau) * var_target
                update_critic_target_expr.append(var_target.assign(new_var))
            update_critic_target_expr = tf.group(*update_critic_target_expr)

        # update actor target operations
        with tf.variable_scope('update_actor_target'):
            update_actor_target_expr = []
            sorted_vars = sorted(
                actor_func_vars,
                key=lambda v: v.name
            )
            sorted_target_vars = sorted(
                target_actor_func_vars,
                key=lambda v: v.name
            )
            # assign actor variables to target actor variables
            for var, var_target in zip(sorted_vars, sorted_target_vars):
                new_var = tau * var + (1 - tau) * var_target
                update_actor_target_expr.append(var_target.assign(new_var))
            update_actor_target_expr = tf.group(*update_actor_target_expr)

        # action theano-style function
        act = util.function(inputs=[obs_t_input], outputs=policy_t)

        # train theano-style function
        train_actor = util.function(
            inputs=[
                obs_t_input
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
        update_actor_target = util.function(
            [],
            [],
            updates=[update_actor_target_expr]
        )
        update_critic_target = util.function(
            [],
            [],
            updates=[update_critic_target_expr]
        )

        return act, train_actor, train_critic, update_actor_target, update_critic_target
