libdl
===

Awesome deep learning library from scratch

## Xor Solver

To train and run the xor sample program run it from the build directory with two binary numbers as parameter.

Exampe:

`build/samples/xor/xor 1 0`

## digit Recognizer

There are 2 variants. The first uses fully connected layer. 
The second uses fully connected layers + convolution layers.
Both example programs require OpenCV to display test images.
The fetch_data script downloads the datasets from http://yann.lecun.com/exdb/mnist/
and extracts them in the current directory. If you download them yourself make sure to
pass the download directory as argument.
Both sample programs train the network until it has an accuracy of 97% and at most for 10 epochs.

Example1:

`> cd build/samples/digits/`
`> ./fetch_data.sh/`
`> ./digits ./`

Example2:

`> cd build/samples/digits_cnn/`
`> ./fetch_data.sh/`
`> ./digits_cnn ./`
