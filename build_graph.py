import tensorflow as tf
import lightsaber.tensorflow.util as util


def build_train(q_func, num_actions, optimizer, batch_size=32,
                grad_norm_clipping=10.0, gamma=1.0, scope='deepq', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_t_input = tf.placeholder(tf.float32, [None, 84, 84, 4], name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_input = tf.placeholder(tf.float32, [None, 84, 84, 4], name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        q_t = q_func(obs_t_input, num_actions, scope='q_func')
        q_func_vars = util.scope_vars(util.absolute_scope_name('q_func'))

        q_tp1 = q_func(obs_tp1_input, num_actions, scope='target_q_func')
        target_q_func_vars = util.scope_vars(util.absolute_scope_name('target_q_func'))

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf.reduce_mean(util.huber_loss(td_error))

        gradients = optimizer.compute_gradients(errors, var_list=q_func_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                    sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        actions = tf.argmax(q_t, axis=1)
        act = util.function(inputs=[obs_t_input], outputs=actions)

        train = util.function(
            inputs=[
                obs_t_input, act_t_ph, rew_t_ph, obs_tp1_input, done_mask_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = util.function([], [], updates=[update_target_expr])

        q_values = util.function([obs_t_input], q_t)

        return act, train, update_target, q_values
