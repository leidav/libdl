#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <memory>

#include <layer/fully_connected_layer.h>
#include <layer/relu_layer.h>
#include <layer/sigmoid_layer.h>
#include <layer/softmax_layer.h>
#include <neural_network.h>

TEST_CASE("Fully Connected Layer", "[FCL]") {
  nn::Layer::Array x(2, 4);
  x << 1, 2, 3, 4, 5, 6, 7, 8;
  nn::Layer::Array expected_y(2, 3);
  expected_y << 2, 3, 4, 6, 7, 8;

  nn::FullyConnectedLayer fcl(2, 4, 3, 1e-6f);
  fcl.forward(x);
  const nn::Layer::Array& y = fcl.y();
  REQUIRE(y.rows() == 2);
  REQUIRE(y.cols() == 3);
  REQUIRE(y.isApprox(expected_y));
}

TEST_CASE("Sigmoid Layer", "[SIG]") {
  nn::Layer::Array x(2, 4);
  x << 1, 2, 3, 4, 5, 6, 7, 8;
  nn::Layer::Array expected_y(2, 4);
  expected_y << 0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f, 0.99330715f,
      0.99752738f, 0.99908895f, 0.99966465f;
  nn::SigmoidLayer sigmoid(2, 4);
  sigmoid.forward(x);
  const nn::Layer::Array& y = sigmoid.y();
  REQUIRE(y.isApprox(expected_y));
  std::cout << "array: " << y << std::endl;
}

TEST_CASE("ReLU Layer", "[ReLU]") {
  nn::Layer::Array x(2, 4);
  x << -1, -0.2f, 0.1f, 4, -5, 6, -7, 8;
  nn::Layer::Array expected_y(2, 4);
  expected_y << 0, 0, 0.1f, 4, 0, 6, 0, 8;
  nn::ReLULayer relu(2, 4);
  relu.forward(x);
  const nn::Layer::Array& y = relu.y();
  REQUIRE(y.isApprox(expected_y));
  std::cout << "array: " << y << std::endl;
}

TEST_CASE("Neural Network", "[NN]") {
  nn::NeuralNetwork net;
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 100, 200, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(32, 200));
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 200, 100, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(32, 100));
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 100, 10, 1e-6f));
  net.addOutputLayer(std::make_unique<nn::SoftmaxLayer>(32, 10));

  nn::Layer::Array a = nn::Layer::Array::Random(32, 100);
  std::vector<int> labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3,
                             4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
                             8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float loss = net.forward(a, labels);
  net.backward(a);
  const nn::Layer::Array& y = net.y();
  REQUIRE(y.rows() == 32);
  REQUIRE(y.cols() == 10);
  std::cout << "loss: " << loss << std::endl;
  std::cout << "array: " << y << std::endl;
}
