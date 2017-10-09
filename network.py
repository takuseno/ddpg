import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens, inpt, num_actions, bound, scope='actor', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = tf.layers.dense(out, hidden, name='d1')
            #out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        out = tf.layers.dense(out, num_actions,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='d2')
        out = tf.nn.tanh(out)
    return out

def _make_critic_network(inpt, action, num_actions, scope='critic', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        #net = tf.layers.dense(inpt, 30, name='d1')
        #net = layers.layer_norm(net, center=True, scale=True)
        #net = tf.nn.relu(net)

        out = tf.concat([inpt, action], axis=1)

        out = tf.layers.dense(out, 30, name='d2')
        #out = layers.layer_norm(out, center=True, scale=True)
        out = tf.nn.relu(out)

        out = tf.layers.dense(out, 1,
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='d3')
    return out

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network():
    return lambda *args, **kwargs: _make_critic_network(*args, **kwargs)
