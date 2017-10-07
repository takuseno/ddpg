import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_actor_network(hiddens, inpt, num_actions, bound, scope='actor', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('linear_layers'):
            for hidden in hiddens:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=tf.nn.tanh)
        scaled_out = tf.mul(out, bound)
        return out, scaled_out

def _make_critic_network(inpt, action, num_actions, scope='critic', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        net = layers.fully_connected(inpt, num_outputs=400, activation_fn='relu')
        t1 = layers.fully_connected(net, num_outputs=300)
        t2 = layers.fully_connected(action, num_outputs=300)
        out = tf.nn.relu(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b)
        out = layers.fully_connected(out, num_outputs=1)
        return out

def make_actor_network(hiddens):
    return lambda *args, **kwargs: _make_actor_network(hiddens, *args, **kwargs)

def make_critic_network():
    return lambda *args, **kwargs: _make_critic_network(*args, **kwargs)
