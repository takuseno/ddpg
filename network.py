import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens, inpt, num_actions, scope='actor', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(out, hidden, name='d1',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
            out = tf.nn.relu(out)
        out = tf.layers.dense(out, num_actions,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='d2')
        out = tf.nn.tanh(out)
    return out

def _make_critic_network(inpt, action, num_actions, scope='critic', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = tf.concat([inpt, action], axis=1)

        out = tf.layers.dense(out, 30, name='d2',
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.3))
        out = tf.nn.relu(out)

        out = tf.layers.dense(out, 1,
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='d3')
    return out

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network():
    return lambda *args, **kwargs: _make_critic_network(*args, **kwargs)
