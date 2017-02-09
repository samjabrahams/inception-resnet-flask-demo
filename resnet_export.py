"""
Loads pre-trained Inception-ResNet V2 and exports frozen graph protobuf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import sys
import tarfile
from six.moves.urllib.request import urlretrieve

import tensorflow as tf

import freeze_graph as freeze
import inception_preprocessing as preprocess
import inception_resnet_v2 as resnet
import optimize_for_inference_lib as optimize


tf.app.flags.DEFINE_string(
    'output', 'inception_resnet',
    'Base pathname for exported metagraph and data files')

tf.app.flags.DEFINE_integer(
    'image_size', 299, 'Expected image input height')

tf.app.flags.DEFINE_string(
    'ckpt_url',
    'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar'
    '.gz',
    'URL of checkpoint file for pretrained Inception ResNet V2 Model')

tf.app.flags.DEFINE_string('data_dir', 'serving/static',
    'Directory where checkpoints and metagraphs are saved')
tf.app.flags.DEFINE_bool('delete_checkpoint', True,
    'If True, deletes downloaded ResNet checkpoint after creating frozen output'
    'file (saves disk space)')

FLAGS = tf.app.flags.FLAGS


def maybe_download_and_extract_checkpoint():
    zipped_basename = os.path.basename(FLAGS.ckpt_url)
    zipped_filename = os.path.join(FLAGS.data_dir, zipped_basename)
    ckpt_basename = zipped_basename[:zipped_basename.find('.')] + '.ckpt'
    ckpt_filename = os.path.join(FLAGS.data_dir, ckpt_basename)
    if os.path.exists(ckpt_filename):
        print('Found checkpoint file in {}.'.format(ckpt_filename))
    else:
        if os.path.exists(zipped_filename):
            print('Found zipped checkpoint file in {}.'.format(zipped_filename))
        else:
            print('Downloading {}.'.format(FLAGS.ckpt_url))
            zipped_filename, _ = urlretrieve(FLAGS.ckpt_url, zipped_filename)
            print('Download complete.')

        print('Extracting {}.'.format(zipped_filename))
        tar = tarfile.open(zipped_filename)
        sys.stdout.flush()
        tar.extractall(FLAGS.data_dir)
        tar.close()
        print('Extraction complete.')
        os.remove(zipped_filename)
    return ckpt_filename


def build_model(graph):
    with graph.as_default():
        with tf.contrib.slim.arg_scope(resnet.inception_resnet_v2_arg_scope()):
            placeholder = tf.placeholder(tf.string, name='input')
            image = tf.image.decode_jpeg(placeholder, channels=3, name='image')
            image = preprocess.preprocess_for_eval(image, FLAGS.image_size,
                                                   FLAGS.image_size)
            image = tf.expand_dims(image, 0)
            logits, end_points = resnet.inception_resnet_v2(
                image, is_training=False, dropout_keep_prob=1.0)
        saver = tf.train.Saver(tf.global_variables())
    return placeholder, logits, end_points


def export_model(graph, ckpt_filename, placeholder, logits, end_points):
    # Export graph definition
    tf.train.write_graph(graph, FLAGS.data_dir, FLAGS.output + '.pb')
    proto_filename = os.path.join(FLAGS.data_dir, FLAGS.output + '.pb')
    output_filename = os.path.join(FLAGS.data_dir, FLAGS.output + '_frozen.pb')

    predictions = end_points['Predictions']

    # Freeze the model
    print('Freezing model.')
    freeze.freeze_graph(input_graph=proto_filename,
                        input_saver='',
                        input_binary=False,
                        input_checkpoint=ckpt_filename,
                        output_node_names=','.join([logits.op.name,
                                                    predictions.op.name]),
                        restore_op_name='save/restore_all',
                        filename_tensor_name='save/Const:0',
                        output_graph=output_filename,
                        clear_devices=True,
                        initializer_nodes='')
    print('Model frozen.')
    frozen_graph_def = tf.GraphDef()
    with open(output_filename, "rb") as f:
        data = f.read()
        frozen_graph_def.ParseFromString(data)

    print('Optimizing model.')
    optimized_graph_def = optimize.optimize_for_inference(
        frozen_graph_def, [placeholder.op.name],
        [logits.op.name, predictions.op.name], tf.string.as_datatype_enum)
    with open(output_filename, 'wb') as f:
        f.write(optimized_graph_def.SerializeToString())
    print('Model optimized.')
    return output_filename, proto_filename


def main(_):
    ckpt_filename = maybe_download_and_extract_checkpoint()
    graph = tf.Graph()
    placeholder, logits, end_points = build_model(graph)
    output_filename, proto_filename = export_model(graph, ckpt_filename, 
                                                   placeholder, logits, 
                                                   end_points)
    print('Inference graph file: {}'.format(output_filename))
    print('Input name: ', placeholder.name)
    print('Logits name: ', logits.name)
    print('Predictions name: ', end_points['Predictions'].name)
    if FLAGS.delete_checkpoint:
        os.remove(ckpt_filename)
        os.remove(proto_filename)


if __name__ == '__main__':
    tf.app.run()
