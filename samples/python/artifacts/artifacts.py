#!/bin/env python
import pydl
from pydl import utils
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from datetime import datetime

image_height=321
image_width=481
tile_size=24
tiles_per_row=image_width//tile_size
rows_of_tiles=image_height//tile_size
tiles_per_image=tiles_per_row*rows_of_tiles
rotations=1
channles=3

def loadImages(name,directory,image_count):
    images=np.empty((image_count*tiles_per_image*rotations,tile_size*tile_size*channles),dtype=np.uint8)
    for i in range(image_count):
       loadImage(images,i,f"{directory}/{i}.jpg") 
    return images

def loadImage(images,row,path):
    image=Image.open(path) 
    i=0
    for y in range(image.height//tile_size):
        for x in range(image.width//tile_size):
            tile=image.crop((x*tile_size,y*tile_size,(x*tile_size)+tile_size,(y*tile_size)+tile_size))
            for r in range(rotations):
                rotated=tile.rotate(r*90)
                array=np.array(rotated,dtype=np.uint8)
                images[row*tiles_per_image*rotations+i]=array.reshape(1,array.size)
                i+=1
    image.close()

def main():
    if len(sys.argv) <2:
        print("too few arguments!")
        print("you have to specify the path to the dataset directory!")
        exit(1)

    directory=sys.argv[1]

    loadfile=""
    if len(sys.argv)>=3:
        loadfile=sys.argv[2]

    print("Loading Images...")

    train_images=loadImages("train_input",f"{directory}/train/input/",400)
    train_ground_truth=loadImages("train_ground_truth",f"{directory}/train/ground_truth/",400)
    test_images=loadImages("test_input",f"{directory}/test/input/",100)
    test_ground_truth=loadImages("test_ground_truth",f"{directory}/test/ground_truth/",100)

    mini_batch_size = 128
    train_size = train_images.shape[0]
    test_size = test_images.shape[0]
    image_size = train_images.shape[1]
    num_train_mini_batches = int(np.ceil(train_size / mini_batch_size))
    num_test_mini_batches = int(np.ceil(test_size / mini_batch_size))

    mini_batch_images=np.zeros((mini_batch_size, image_size),dtype=np.float32)
    mini_batch_ground_truth=np.zeros((mini_batch_size, image_size),dtype=np.float32)

    print(f"num train images: {train_size}")
    print(f"num test images: {test_size}")

    net = pydl.NeuralNetwork(mini_batch_size)

    time=datetime.now() 
    net.openSaveFile(f"{time.year}-{time.month}-{time.day}-{time.hour}-{time.minute}-{time.second}.save",0)

    l2_regularization = 5e-6
    
    input_width=tile_size
    input_height=tile_size

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*channles))
    kernel=9
    padding=4
    stride=1
    net.addHiddenLayer(pydl.ConvolutionLayer(input_width,input_height,channles,64,kernel,mini_batch_size,padding,stride,l2_regularization))
    iput_width,input_height=pydl.utils.convolution_helper.convolutionOutputSize(input_width,input_height,kernel,padding,stride)

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*64))

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*64))

    #net.addHiddenLayer(pydl.Convolution1x1Layer(input_width,input_height,64,32,mini_batch_size,l2_regularization))

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*32))

    #net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*32))

    kernel=7
    padding=3
    stride=1
    net.addHiddenLayer(pydl.ConvolutionLayer(input_width,input_height,64,32,kernel,mini_batch_size,padding,stride,l2_regularization))
    iput_width,input_height=pydl.utils.convolution_helper.convolutionOutputSize(input_width,input_height,kernel,padding,stride)

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*32))

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*32))

    net.addHiddenLayer(pydl.Convolution1x1Layer(input_width,input_height,32,16,mini_batch_size,l2_regularization))

    #net.addHiddenLayer(pydl.BatchnormLayer(mini_batch_size,input_width*input_height*16))

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*16))

    kernel=5
    padding=2
    stride=1
    net.addHiddenLayer(pydl.ConvolutionLayer(input_width,input_height,16,channles,kernel,mini_batch_size,padding,stride,l2_regularization))
    iput_width,input_height=pydl.utils.convolution_helper.convolutionOutputSize(input_width,input_height,kernel,padding,stride)
    #net.addResidual(0)

    net.addHiddenLayer(pydl.LeakyReLULayer(input_width*input_height*channles))

    net.addOutputLayer(pydl.LeastSquaresLayer(input_width*input_height*channles))

    if len(loadfile)>0:
        net.loadLastParameters(loadfile)

    print("Start Training...")

    for epoch in range(10):
        print(f"start epoch {epoch}")
        print("train")
        train_loss=0.0
        permutated_indices=np.random.permutation(train_size)
        for i in range(num_train_mini_batches):
            indices=permutated_indices[np.arange(i*mini_batch_size,(i+1)*mini_batch_size)%train_size]
            mini_batch_images[np.arange(mini_batch_size)]=train_images[indices]
            mini_batch_ground_truth[np.arange(mini_batch_size)]=train_ground_truth[indices]
            mini_batch_images/=255.0
            mini_batch_ground_truth/=255.0
            loss=net.forward(mini_batch_images,mini_batch_ground_truth,True)
            train_loss+=loss
            net.backward(mini_batch_images,10e-4)
            print(f"train batch {i} of {num_train_mini_batches}, loss:{loss},psnr:{10*np.log10(1.0/loss)}")
            if i % 10 == 0:
                net.saveParameters(loss,np.finfo(np.float32).max)
        train_loss/=num_train_mini_batches

        print("test")
        test_loss=0.0
        accuracy=0.0
        for i in range(num_test_mini_batches):
            indices=np.arange(i*mini_batch_size,(i+1)*mini_batch_size)%test_size
            mini_batch_images[np.arange(mini_batch_size)]=test_images[indices]
            mini_batch_ground_truth[np.arange(mini_batch_size)]=test_ground_truth[indices]
            mini_batch_images/=255.0
            mini_batch_ground_truth/=255.0
            loss=net.forward(mini_batch_images,mini_batch_ground_truth,False)
            test_loss+=loss;
            print(f"test batch {i} of {num_test_mini_batches}, loss:{loss}")
        test_loss/=num_test_mini_batches
        print(f"epoch:{epoch}, train loss:{train_loss}, "
              f"test_loss: {test_loss}")
        net.saveParameters(train_loss,test_loss)
         
if __name__=='__main__':
    main()
