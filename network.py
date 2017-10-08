import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens, inpt, num_actions, bound, scope='actor', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('linear_layers'):
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=tf.nn.tanh,
            weights_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        scaled_out = tf.multiply(out, bound)
        return scaled_out

def _make_critic_network(inpt, action, num_actions, scope='critic', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        net = layers.fully_connected(inpt, num_outputs=400, activation_fn=tf.nn.relu)
        out = tf.concat([net, action], axis=-1)
        net = layers.fully_connected(out, num_outputs=300, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=1,
            weights_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return out

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network():
    return lambda *args, **kwargs: _make_critic_network(*args, **kwargs)
