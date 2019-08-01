#!/bin/env python
import pydl
from pydl import utils
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

image_height=321
image_width=481
channels=3

def loadImages(name,directory,image_number):
    images=np.empty((1,image_width*image_height*channels),dtype=np.uint8)
    print(image_number)
    width,height=loadImage(images,f"{directory}/{image_number}.jpg") 
    return (images,width,height)

def loadImage(images,path):
    image=Image.open(path) 
    array=np.array(image,dtype=np.uint8)
    images[0]=array.reshape(1,array.size)
    image.close()
    return (image.width,image.height)

def main():
    if len(sys.argv) <2:
        print("too few arguments!")
        print("you have to specify the path to the dataset directory!")
        exit(1)

    directory=sys.argv[1]

    loadfile=""
    if len(sys.argv)>=3:
        loadfile=sys.argv[2]

    img=0
    if len(sys.argv)>=4:
        img=int(sys.argv[3])

    print("Loading Images...")
    test_images,width,height=loadImages("test_input",f"{directory}/test/input/",img)
    test_ground_truth,_,_=loadImages("test_input",f"{directory}/test/ground_truth/",img)

    mini_batch_size = 1
    image_size = test_images.shape[1]

    l2_regularization = 1e-5
    mini_batch_images=np.zeros((mini_batch_size, image_size),dtype=np.float32)
    mini_batch_ground_truth=np.zeros((mini_batch_size, image_size),dtype=np.float32)
    net = pydl.NeuralNetwork(mini_batch_size)
    
    input_width=width
    input_height=height

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*channles))
    kernel=9
    padding=4
    stride=1
    net.addHiddenLayer(pydl.ConvolutionLayer(input_width,input_height,channels,64,kernel,mini_batch_size,padding,stride,l2_regularization))
    input_width,input_height=pydl.utils.convolution_helper.convolutionOutputSize(input_width,input_height,kernel,padding,stride)

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*64))

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*64))

    #net.addHiddenLayer(pydl.Convolution1x1Layer(input_width,input_height,64,32,mini_batch_size,l2_regularization))

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*32))

    #net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_width*32))

    kernel=7
    padding=3
    stride=1
    net.addHiddenLayer(pydl.ConvolutionLayer(input_width,input_height,64,32,kernel,mini_batch_size,padding,stride,l2_regularization))
    input_width,input_height=pydl.utils.convolution_helper.convolutionOutputSize(input_width,input_height,kernel,padding,stride)

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*32))

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*32))

    net.addHiddenLayer(pydl.Convolution1x1Layer(input_width,input_height,32,16,mini_batch_size,l2_regularization))

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*16))

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*16))

    kernel=5
    padding=2
    stride=1
    net.addHiddenLayer(pydl.ConvolutionLayer(input_width,input_height,16,channels,kernel,mini_batch_size,padding,stride,l2_regularization))
    input_width,input_height=pydl.utils.convolution_helper.convolutionOutputSize(input_width,input_height,kernel,padding,stride)
    #net.addResidual(0)

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*channels))

    net.addOutputLayer(pydl.LeastSquaresLayer(input_width*input_height*channels))

    if len(loadfile)>0:
        net.loadLastParameters(loadfile)
        #net.loadBestParameters(loadfile)

    print("Start...")

    mini_batch_images[0]=test_images[0]
    mini_batch_ground_truth[0]=test_ground_truth[0]
    mini_batch_images/=255.0
    mini_batch_ground_truth/=255.0
    loss=net.forward(mini_batch_images,mini_batch_ground_truth,False)

    print(f"loss: {loss}")

    fig=plt.figure(figsize=(20,20))
    fig.add_subplot(2,1,1)
    plt.title("input")
    plt.imshow(mini_batch_images[0].reshape(height,width,channels))
    fig.add_subplot(2,1,2)
    plt.title("output")
    plt.imshow(net.result()[0].reshape(height,width,channels))
    #fig.add_subplot(2,2,3)
    #plt.title("ground truth")
    #plt.imshow(mini_batch_ground_truth[0].reshape(height,width,3))
    plt.show()
         
if __name__=='__main__':
    main()
