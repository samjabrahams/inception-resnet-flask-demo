# Pretrained Inception-ResNet Demo with Flask

## Quick note

This demo is _not_ supposed to represent a production-ready server. It is not secure for the web, and is not setup to be responsive (server-client responses and model evaluation run in the same thread instead of asyncronously). Rather, this is a quick demonstration of how to utilize a pretrained Inception-ResNet model and quickly put together a prototype with it.

## Dependencies

* [NumPy](http://www.numpy.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Flask](http://flask.pocoo.org/)
* [werkzeug](http://werkzeug.pocoo.org/)

## Run it!

1. Make sure you have the above dependencies installed (`pip install ...`)
2. Clone the repo, move into the directory, and execute `run_server.sh`:

```
$ git clone https://github.com/samjabrahams/inception-resnet-flask-demo.git
$ cd inception-resnet-flask-demo
$ ./run_server.sh
```

## About the files

### `run_server.sh`

* The main file, runs the preprocessing of files and then launches the Flask server
* You must run this script from its directory, not from a parent or child directory. ie. this:
```
$ ./run_server.sh
```
not:
```
$ ../run_server.sh
```
or:
```
$ inception-resnet-flask-demo/run_server.sh
```

### Pre-processing files

#### `resnet_export.py`

* Downloads Inception-ResNet model and checkpoint files and converts them into a frozen and optimized protobuf file in `serving/static`

#### `labels/merge.py`

* Creates `descriptions.txt`, which is a line separated file that provides text descriptions of the outputs of the pretrained Inception-ResNet modeel. The first line corresponds to the output at index 0, the second line corresponds to the output at index 1 etc.

### Flask Server files

#### `serving/serving.py`

* Starts up the Flask server and TensorFlow model Session
* Contains route functions for the server

#### `serving/model.py`

* Functions and singleton Session class for the Inception-ResNet Model
* Singleton is used to preserve optimizations and prevent users from reloading the model from memory

#### `serving/templates/layout.html` and `serving/templates/predict.html`

* Jinja2 template files for the basic client interface.

--- 

### TensorFlow files.

The [TENSORFLOW_LICENSE](TENSORFLOW_LICENSE) applies to the following files:

#### `inception_resnet_v2.py`

* Constructs the Inception ResNet V2 model
* From the [TensorFlow Models repository](https://github
.com/tensorflow/models/blob/master/slim/nets/inception_resnet_v2.py)

#### `inception_preprocessing.py`

* Creates image steps to preprocess image data for both training and
inference
* [From the TensorFlow Models repository](https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py)

#### `freeze_graph.py`

* Tool for "freezing" a TensorFlow graph. Converts Variable objects into constants.
* [From the TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)

#### `optimize_for_inference.py`

* Tool that removes unnecessary operations from a graph.
* [From the TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py)

#### `optimize_for_inference_lib.py`

* Utility functions needed for `optimize_for_inference.py`
* [From the TensorFlow repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_lib.py)
