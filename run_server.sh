#!/bin/bash

# Create label descriptions
if [ ! -f serving/static/descriptions.txt ]; then
    echo "Creating label descriptions file.."
    cd labels
    python3 merge.py
    cp descriptions.txt ../serving/static/descriptions.txt
    cd ..
    echo "Done."
fi

# Load, freeze, optimize, and export ResNet model
if [ ! -f serving/static/inception_resnet_frozen.pb ]; then
    echo "Creating frozen ResNet model file.."
    python3 resnet_export.py
    echo "Done"
fi

# Create temp directory if necessary
mkdir -p serving/temp

# Serve Flask application
echo "Launching server."
cd serving
export FLASK_APP=serving.py
flask run
