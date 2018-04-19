import tensorflow as tf
import numpy as np


# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=500,
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out


class NNDynamicsModel:
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.sess = sess

        #training hyperparameters
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate

        # Define input which is states and actions, placeholders are only for unnormalized
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.sy_obs_un = tf.placeholder(tf.float32, shape=[None, obs_dim], name='obs_un')
        self.sy_act_un = tf.placeholder(tf.float32, shape=[None, act_dim], name='act_un')
        self.sy_next_obs_un = tf.placeholder(tf.float32, shape=[None, obs_dim], name='next_obs_un')

        mean_obs, std_obs, mean_deltas, std_deltas, mean_action, std_action = normalization
        episilon = 1e-8
        sy_obs = (self.sy_obs_un - mean_obs) / (std_obs + episilon)
        sy_act = (self.sy_act_un - mean_action) / (std_action + episilon)
        # sy_next_obs = (self.sy_next_obs_un - mean_obs) / (std_obs + episilon)

        # Concatenate to simulate f(s,a)
        sy_in = tf.concat([sy_obs, sy_act], axis=1)

        # Define model
        sy_out_un = build_mlp(sy_in, obs_dim, 'nn', n_layers, size, activation, output_activation)
        self.sy_out = mean_deltas + std_deltas * sy_out_un + self.sy_obs_un

        self.loss = tf.losses.mean_squared_error(self.sy_next_obs_un, self.sy_out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states
        and fit the dynamics model going from normalized states, normalized actions
        to normalized state differences (s_t+1 - s_t)
        """

        obs = np.concatenate([data[i]['obs'] for i in range(len(data))])
        act = np.concatenate([data[i]['act'] for i in range(len(data))])
        next_obs = np.concatenate([data[i]['next_obs'] for i in range(len(data))])

        qtt_batches = int(np.ceil(len(obs) / self.batch_size))

        losses = []
        for i in range(self.iterations):
            loss_avg_batches = 0
            for x in range(qtt_batches):
                start = x * self.batch_size
                end = (x+1) * self.batch_size
                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.sy_obs_un: obs[start:end],
                                                                                self.sy_act_un: act[start:end],
                                                                                self.sy_next_obs_un: next_obs[start:end]})
                loss_avg_batches += loss
            loss_avg_batches /= qtt_batches
            losses.append(loss_avg_batches)
        return losses

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions
        and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        return self.sess.run(self.sy_out, feed_dict={self.sy_obs_un: states, self.sy_act_un: actions})
