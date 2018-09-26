import numpy as np

class policy(object):
    def __init__(self, args, infos, **kwargs):
        self.args = args
        self.infos = infos
        self._unparsed = kwargs

        self.input_ph = {}
        self.tensor = {}
        self.update_op = {}

        self.npr = np.random.RandomState(args.seed)

    def _build_model(self):
        raise NotImplementedError

    def act(self, data_dict):
        raise NotImplementedError

    def train(self, data_dict):
        raise NotImplementedError