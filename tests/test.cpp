#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <memory>

#include <layer/dropout_layer.h>
#include <layer/fully_connected_layer.h>
#include <layer/least_squares_layer.h>
#include <layer/relu_layer.h>
#include <layer/sigmoid_layer.h>
#include <layer/softmax_layer.h>
#include <neural_network.h>

#include <update_rule.h>

TEST_CASE("Fully Connected Layer", "[FCL]") {
  nn::Layer::Array x(2, 4);
  x << 1, 2, 3, 4, 5, 6, 7, 8;
  nn::Layer::Array expected_y(2, 3);
  expected_y << 2, 3, 4, 6, 7, 8;

  nn::FullyConnectedLayer fcl(2, 4, 3, 1e-6f);
  fcl.forward(x, false);
  const nn::Layer::Array& y = fcl.y();
  REQUIRE(y.rows() == 2);
  REQUIRE(y.cols() == 3);
  // REQUIRE(y.isApprox(expected_y));

  // std::cout << "array: " << y << std::endl;
}

TEST_CASE("Sigmoid Layer", "[SIG]") {
  nn::Layer::Array x(2, 4);
  x << 1, 2, 3, 4, 5, 6, 7, 8;
  nn::Layer::Array expected_y(2, 4);
  expected_y << 0.73105858f, 0.88079708f, 0.95257413f, 0.98201379f, 0.99330715f,
      0.99752738f, 0.99908895f, 0.99966465f;
  nn::SigmoidLayer sigmoid(2, 4);
  sigmoid.forward(x, false);
  const nn::Layer::Array& y = sigmoid.y();
  REQUIRE(y.isApprox(expected_y));
}

TEST_CASE("ReLU Layer", "[ReLU]") {
  nn::Layer::Array x(2, 4);
  x << -1, -0.2f, 0.1f, 4, -5, 6, -7, 8;
  nn::Layer::Array expected_y(2, 4);
  expected_y << 0, 0, 0.1f, 4, 0, 6, 0, 8;
  nn::ReLULayer relu(2, 4);
  relu.forward(x, false);
  const nn::Layer::Array& y = relu.y();
  REQUIRE(y.isApprox(expected_y));
}
TEST_CASE("Dropout Layer", "[Dropout]") {
  nn::Layer::Array x = nn::Layer::Array::Ones(4, 100);
  nn::DropOutLayer dropout(4, 100);
  dropout.forward(x, true);
  const nn::Layer::Array& y = dropout.y();
  nn::Layer::Array row_sum = y.rowwise().sum();
  float average = row_sum.sum() / y.rows();
  std::cout << "average dropout: " << average << std::endl;
  float tollerance = (y.cols() / 100.0f) * 10.0f;
  float min = y.cols() / 2.0f - tollerance;
  float max = y.cols() / 2.0f + tollerance;
  REQUIRE(average > min);
  REQUIRE(average < max);
}

TEST_CASE("Neural Network", "[NN]") {
  nn::NeuralNetwork net;
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 100, 200, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(32, 200));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(32, 200));
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 200, 100, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(32, 100));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(32, 100));
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 100, 10, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(32, 10));
  net.addOutputLayer(std::make_unique<nn::SoftmaxLayer>(32, 10));

  nn::Layer::Array a = nn::Layer::Array::Random(32, 100);
  Eigen::VectorXf labels(40, 1);
  labels << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
      3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  float loss = net.forward(a, nn::Layer::Array(labels), true);
  // net.backward(a);
  const nn::Layer::Array& y = net.y();
  REQUIRE(y.rows() == 32);
  REQUIRE(y.cols() == 10);
  std::cout << "loss: " << loss << std::endl;
  std::cout << "array: " << y << std::endl;
}

TEST_CASE("Xor Test", "[Xor]") {
  nn::NeuralNetwork net;
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(4, 2, 16, 1e-8f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(4, 16));
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(4, 16, 1, 1e-8f));
  net.addOutputLayer(std::make_unique<nn::LeastSquaresLayer>(4, 1));

  nn::Layer::Array input = nn::Layer::Array(4, 2);
  nn::Layer::Array expected_output = nn::Layer::Array(4, 1);
  input << 0, 0, 0, 1, 1, 0, 1, 1;
  expected_output << 0, 1, 1, 0;

  // train 1000 epochs
  float loss = 0;
  for (int i = 0; i < 1000; i++) {
	loss = net.forward(input, expected_output, true);
	net.backward(input, 0.01);
  }
  net.execute(input);
  nn::Layer::Array output = net.y().round();
  REQUIRE(output.isApprox(expected_output));
  std::cout << "loss: " << loss << std::endl;
  std::cout << "array: " << output + 0.0 << std::endl;
}
