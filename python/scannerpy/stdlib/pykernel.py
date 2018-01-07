from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

class TensorFlowKernel:
    def __init__(self, protobufs, config):
        # TODO: wrap this in "with device"
        config = tf.ConfigProto(allow_soft_placement = True)
        self.graph = self.build_graph()
        self.sess = tf.Session(graph=self.graph, config=config)

    def close(self):
        self.sess.close()

    def build_graph(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
