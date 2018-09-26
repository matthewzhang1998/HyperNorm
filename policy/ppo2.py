import numpy as np
import tensorflow as tf

from . import base_policy
from util.tf_networks import feed_forward
from util.tf_distributions import multivariate_gaussian
from util.norm_util import build_norm


class policy(base_policy.policy):
    def __init__(self, args, infos, **kwargs):
        super(policy, self).__init__(args, infos, **kwargs)
        self._build_model()

    def _build_model(self):
        self._build_preprocessing()
        self._build_networks()
        self._build_loss()

    def _build_preprocessing(self):
        self.whitening = build_norm(self.infos['obs_size'], self.args)

        self.input_ph['start_state'] = tf.placeholder(
            dtype=tf.float32, shape=[None, self.infos['obs_size']],
            name='start_state'
        )
        self.input_ph['action'] = tf.placeholder(
            dtype=tf.float32, shape=[None, self.infos['act_size']],
            name='action'
        )
        self.input_ph['advantage'] = tf.placeholder(
            dtype=tf.float32, shape=[None,], name='advantage'
        )

        self.input_ph['value_target'] = tf.placeholder(
            dtype = tf.float32, shape = [None,], name='value_target'
        )
        self.input_ph['log_p_action'] = tf.placeholder(
            dtype=tf.float32, shape=[None,], name = 'log_p'
        )
        self.input_ph['learning_rate'] = tf.placeholder(
            dtype=tf.float32, shape=[], name = 'learning_rate'
        )

        self.tensor['normalized_state'] = \
            (self._input_ph['start_state'] -
            self.whitening['state_norm']) / \
            self.whitening['state_std']

    def _build_networks(self):
        network_shape = [self.observation_size] + \
            self.args.policy_network_size + [self.infos['act_size']]
        num_layer = len(network_shape) - 1
        act_type = [self.args.policy_act_type] * (num_layer - 1) + ['none']
        norm_type = [self.args.policy_norm_type] * (num_layer - 1) + ['none']

        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01

        self.policy_network = feed_forward(
            dims = network_shape, acts = act_type,
            norms = norm_type, init_data = init_data
        )

        self.tensor['action_mu'] = self.policy_network(
            self.tensor['normalized_state']
        )
        self.tensor['action_logsigma'] = tf.Variable(
            (0 * self.npr.randn(1, self.infos['act_size'])).astype(np.float32),
            name="action_logsigma", trainable=True
        )

        self.action_distribution = multivariate_gaussian(
            self.tensor['action_mu'], self.tensor['action_logsigma']
        )
        self.tensor['action']

        network_shape = [self.observation_size] + \
                        self.args.value_network_size + [1]
        num_layer = len(network_shape) - 1
        act_type = [self.args.value_act_type] * (num_layer - 1) + ['none']
        norm_type = [self.args.value_norm_type] * (num_layer - 1) + ['none']

        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )

        self.value_network = feed_forward(
            dims = network_shape, acts = act_type,
            norms = norm_type, init_data = init_data
        )
        self.tensor['value_new'] = self.value_network(
            self.tensor['normalized_state']
        )



    def _build_loss(self):

    def act(self, data_dict):

    def train(self, data_dict):

    def preprocess_data(self, data_dict):


    def generate_advantage(self, data_dict):