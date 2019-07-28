#!/bin/env python
import pydl
from pydl import utils
import sys
import numpy as np
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) <2:
        print("too few arguments!")
        print("you have to specify the path to the dataset directory!")
        exit(1)

    directory=sys.argv[1]

    train_images=utils.mnist.loadImages(directory + "/train-images-idx3-ubyte")
    train_labels=utils.mnist.loadLabels(directory + "/train-labels-idx1-ubyte")
    test_images=utils.mnist.loadImages(directory + "/t10k-images-idx3-ubyte")
    test_labels=utils.mnist.loadLabels(directory + "/t10k-labels-idx1-ubyte")

    mini_batch_size = 30
    train_size = train_images.shape[0]
    test_size = test_images.shape[0]
    image_size = train_images.shape[1]
    num_train_mini_batches = int(np.ceil(train_size / mini_batch_size))
    num_test_mini_batches = int(np.ceil(test_size / mini_batch_size))

    l2_regularization = 1e-5
    mini_batch_images=np.zeros((mini_batch_size, image_size))
    mini_batch_labels=np.zeros((mini_batch_size, 1))
    net = pydl.NeuralNetwork(mini_batch_size)
    net.openSaveFile("test.save",0)

    net.addHiddenLayer(pydl.FullyConnectedLayer(image_size, 1024, l2_regularization))
    net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size, 1024))
    net.addHiddenLayer(pydl.ReLULayer(1024))

    net.addHiddenLayer(pydl.FullyConnectedLayer(1024, 256, l2_regularization))
    net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size, 256))
    net.addHiddenLayer(pydl.ReLULayer(256))

    net.addHiddenLayer(pydl.FullyConnectedLayer(256, 128, l2_regularization))
    net.addHiddenLayer(pydl.ReLULayer(128))
    net.addHiddenLayer(pydl.DropOutLayer(mini_batch_size, 128))

    net.addHiddenLayer(pydl.FullyConnectedLayer(128, 32, l2_regularization))
    net.addHiddenLayer(pydl.ReLULayer(32))
    net.addHiddenLayer(pydl.DropOutLayer(mini_batch_size, 32))

    net.addHiddenLayer(pydl.FullyConnectedLayer(32, 10, l2_regularization))

    net.addOutputLayer(pydl.SoftmaxLayer(10))

    print("Start Training...")

    for epoch in range(10):

        #train 
        train_loss=0.0
        permutated_indices=np.random.permutation(train_size)
        for i in range(num_train_mini_batches):
            indices=permutated_indices[np.arange(i*mini_batch_size,(i+1)*mini_batch_size)%test_size]
            mini_batch_images[np.arange(mini_batch_size)]=train_images[indices]
            mini_batch_labels[np.arange(mini_batch_size)]=train_labels[indices]
            train_loss+=net.forward(mini_batch_images,mini_batch_labels,True)
            net.backward(mini_batch_images,1e-4)
        train_loss/=num_train_mini_batches

        #test
        test_loss=0.0
        accuracy=0.0
        for i in range(num_test_mini_batches):
            indices=np.arange(i*mini_batch_size,(i+1)*mini_batch_size)%test_size
            mini_batch_images[np.arange(mini_batch_size)]=test_images[indices]
            mini_batch_labels[np.arange(mini_batch_size)]=test_labels[indices]
            test_loss+=net.forward(mini_batch_images,mini_batch_labels,False)
            labels=net.y().argmax(axis=1)#.reshape(mini_batch_size,1)
            accuracy+=((labels==mini_batch_labels.transpose()[0]).astype(float)).sum()
        test_loss/=num_test_mini_batches
        accuracy = (accuracy/test_size)*100
        print(f"epoch:{epoch}, accuracy:{accuracy}, train loss:{train_loss}, "
              f"test_loss: {test_loss}")
        net.saveParameters(train_loss,test_loss)
        if accuracy>97:
            break

    print("show 10 random test images and their value\n");
    indices=np.random.randint(0,test_size,mini_batch_size)
    mini_batch_images[np.arange(mini_batch_size)]=test_images[indices]
    mini_batch_labels[np.arange(mini_batch_size)]=test_labels[indices]
    net.forward(mini_batch_images,mini_batch_labels,False)
    results=net.y().argmax(axis=1)
    plt.gray()
    for i in range(10):
        result=results[i]
        truth=int(mini_batch_labels[i][0])
        print(f"result: {result}, ground truth: {truth}");
        plt.imshow(mini_batch_images[i].reshape(28,28))
        plt.show()
         
if __name__=='__main__':
    main()
