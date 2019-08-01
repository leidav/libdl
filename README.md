libdl
====

A Small deep learning library in c++ with python bindings

## Build Instructions

`> mkdir build`

`> cd build`

`> cmake -DCMAKE_BUILD_TYPE=Release`

`> make`

## Generating the Doxygen documentation

`> make doxygen`

The doxygen documentation can be found in doc/html inside the build dirctory.

## Simple Python Api Example

The current example is as simple as possible and not really how you should train a network.
```python
import pydl
from pydl import utils
import numpy as np

batch_size = 30
image_size = 28*28*1
l2_regularization = 1e-5

images=utils.mnist.loadImages("images.bin")
labels=utils.mnist.loadLabels("labels.bin")

net = pydl.NeuralNetwork(batch_size)
net.openSaveFile("test.save",0)

net.addHiddenLayer(pydl.FullyConnectedLayer(image_size, 1024, l2_regularization))

net.addHiddenLayer(pydl.ReLULayer(1024))

net.addHiddenLayer(pydl.FullyConnectedLayer(1024, 10, l2_regularization))

net.addHiddenLayer(pydl.ReLULayer(10))

net.addOutputLayer(pydl.SoftmaxLayer(10))

loss=net.forward(images[np.arange(batch_size)],labels[np.arange(batch_size),True])

net.backward(images[np.arange(batch_size)],1e-4)

net.saveParameters(loss,loss)

result=net.result()

```

## Sample Programs
The sample dirctory contains sample programs using the c++ and python api.

### Xor Solver

To train and run the xor sample program run it from the build directory with two binary numbers as parameter.

Exampe:

`build/samples/xor/xor 1 0`

### MNIST digit recognition

There are 3 variants. The first uses fully connected layer.
The second uses fully connected layers + convolution layers.
The third one is implemented in python and uses fully connected layers.
The c++ variants only output the result and the ground truth label.
The python variant shows the image using pyplot.
The fetch_data script downloads the datasets from http://yann.lecun.com/exdb/mnist/
and extracts them in the current directory. If you download them yourself make sure to
pass the download directory as argument.
All sample programs train the network until it has an accuracy of 97% and at most for 10 epochs. If you wonder why the test loss is smaler than the train loss. Its because of the 50% dropout during training.
Before executing the python version you have to set the PYTHONPATH enviroment variable to the bindings build directory.
This can be don by sourcing enviroment.sh

Example1:

`> cd build/samples/digits/`

`> ./fetch_data.sh/`

`> ./digits ./`

Example2:

`> cd build/samples/digits_cnn/`

`> ./fetch_data.sh/`

`> ./digits_cnn ./`

Example3:

`> cd build/samples/python/digits/`

`> ./fetch_data.sh/`

`> source enviroment.sh`

`> ./digits.py ./`

### Final project: compression artifact removal
The fetch and prepare script requires ImageMagics mogrify command to compress the images.
The artifacty.py script creates a new save file with the current date and time in its name.
Every 10 minibatches it appends the networks trainable paramaters and the current loss to it.
The save file can later be used as starting point for a new artifacty.py invocation or for displaying
a test image with artifact_single_image.py. The script opens the image you specified by the image number.
run.sh is a wrapper around artifact_single_image that uses parameter.save.

Preparation:
`> cd build/samples/python/artifact/`

`> ./fetch_and_prepare_data.sh/`

`> source enviroment.sh`

Train from scratch:

`> ./artifact.py ./`

Train from the existing save file:

`> ./artifact.py ./ parameter.save`

Show test image by using parameter.save and image 8:

`> ./run.sh 8`

Show test image by using any save file and image 8:

`> ./artifact_single_image.py ./ name_of_save_file.save 8`
