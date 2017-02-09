import numpy as np
import tensorflow as tf


################################################
# Operation/Tensor getters
################################################
def get_input(sess):
    """Returns a handle to the raw image bytes input placeholder"""
    return sess.graph.get_tensor_by_name('input:0')


def get_image(sess):
    """Returns a handle to the decoded JPEG image Tensor"""
    return sess.graph.get_tensor_by_name('image:0')


def get_logits(sess):
    """Returns a handle to the output logits Tensor"""
    return sess.graph.get_tensor_by_name(
        'InceptionResnetV2/Logits/Logits/BiasAdd:0')


def get_predictions(sess):
    """Returns a handle to the output predictions Tensor"""
    return sess.graph.get_tensor_by_name(
        'InceptionResnetV2/Logits/Predictions:0')


class Session(object):
    """Singleton TensorFlow Session

    Loads in the Inception-ResNet model and performs 10 warmup runs
    """

    __instance = None

    def __new__(cls):
        if Session.__instance is None:
            print('creating singleton')

            # Import graph from protobuf file
            graph = tf.Graph()
            graph_def = tf.GraphDef()
            with graph.as_default():
                with open('static/inception_resnet_frozen.pb', 'rb') as f:
                    graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
            sess = tf.Session(graph=graph)
            print('Session created')

            # Warm up Session for on-the-fly optimizations
            print('Warming up Session')
            dummy_image = np.random.normal(0, 1, [299, 299, 3])
            feed_dict = {get_image(sess): dummy_image}
            for i in range(10):
                _ = sess.run(get_predictions(sess), feed_dict)

            print('Session ready to serve')
            Session.__instance = sess
        return Session.__instance
