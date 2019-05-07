#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <memory>

#include <fully_connected_layer.h>
#include <neural_network.h>
#include <relu_layer.h>
#include <sigmoid_layer.h>
#include <softmax_layer.h>

TEST_CASE("Neural Network", "[NN]") {
  nn::NeuralNetwork net;
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(
      nn::FullyConnectedLayer(32, 100, 200)));
  net.addHiddenLayer(
      std::make_unique<nn::SigmoidLayer>(nn::SigmoidLayer(32, 200)));
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(
      nn::FullyConnectedLayer(32, 200, 100)));
  net.addHiddenLayer(
      std::make_unique<nn::SigmoidLayer>(nn::SigmoidLayer(32, 100)));
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(
      nn::FullyConnectedLayer(32, 100, 10)));
  net.addOutputLayer(
      std::make_unique<nn::SoftmaxLayer>(nn::SoftmaxLayer(32, 10)));
  nn::Layer::Array a = nn::Layer::Array::Random(32, 100);
  Eigen::VectorXi labels(40);
  labels << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
      3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  net.forward(a);
  const nn::Layer::Array& y = net.y();
  REQUIRE(y.rows() == 32);
  REQUIRE(y.cols() == 10);
  std::cout << net.loss(labels) << std::endl;
}
