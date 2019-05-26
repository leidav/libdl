#include <layer/fully_connected_layer.h>
#include <layer/least_squares_layer.h>
#include <layer/relu_layer.h>
#include <neural_network.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 3) {
	std::cerr << "too few arguments!" << std::endl;
	std::cerr << "example usage: xor 1 0" << std::endl;
	std::exit(1);
  }
  nn::Layer::Array input(1, 2);
  input << std::atof(argv[1]), std::atof(argv[2]);

  nn::NeuralNetwork net(4);
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(2, 8, 1e-8f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(16));
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(16, 1, 1e-8f));
  net.addOutputLayer(std::make_unique<nn::LeastSquaresLayer>(1));

  nn::Layer::Array train_input = nn::Layer::Array(4, 2);
  nn::Layer::Array expected_output = nn::Layer::Array(4, 1);
  train_input << 0, 0, 0, 1, 1, 0, 1, 1;
  expected_output << 0, 1, 1, 0;

  std::cout << "train 10000 epochs..." << std::endl;

  float loss = 0;
  for (int i = 0; i < 10000; i++) {
	loss = net.forward(train_input, expected_output, true);
	net.backward(train_input, 0.01f);
  }
  std::cout << "train loss: " << loss << std::endl;

  net.inference(input);
  std::cout << "input: " << input(0, 0) << "," << input(0, 1) << std::endl;
  nn::Layer::Array output = net.inference_result().round();
  std::cout << "result: " << output + 0.0f << std::endl;
  return 0;
}
