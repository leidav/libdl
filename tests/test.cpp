#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>
#include <memory>

#include <layer/batchnorm_layer.h>
#include <layer/convolution_layer.h>
#include <layer/dropout_layer.h>
#include <layer/fully_connected_layer.h>
#include <layer/least_squares_layer.h>
#include <layer/relu_layer.h>
#include <layer/sigmoid_layer.h>
#include <layer/softmax_layer.h>
#include <neural_network.h>
#include <utils/convolution_helper/convolution_helper.h>

#include <update_rule.h>

TEST_CASE("Fully Connected Layer", "[FCL]") {
  nn::Layer::Array x(2, 4);
  x << 1, 2, 3, 4, 5, 6, 7, 8;
  nn::Layer::Array expected_y(2, 3);
  expected_y << 2, 3, 4, 6, 7, 8;
  nn::Layer::Array y(2, 3);

  nn::FullyConnectedLayer fcl(4, 3, 1e-6f);
  fcl.forward(y, x, false);
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
  nn::Layer::Array y(2, 4);
  nn::SigmoidLayer sigmoid(4);
  sigmoid.forward(y, x, false);
  REQUIRE(y.isApprox(expected_y));
}

TEST_CASE("ReLU Layer", "[ReLU]") {
  nn::Layer::Array x(2, 4);
  x << -1, -0.2f, 0.1f, 4, -5, 6, -7, 8;
  nn::Layer::Array expected_y(2, 4);
  expected_y << 0, 0, 0.1f, 4, 0, 6, 0, 8;
  nn::Layer::Array y(2, 4);
  nn::ReLULayer relu(4);
  relu.forward(y, x, false);
  REQUIRE(y.isApprox(expected_y));
}
TEST_CASE("Dropout Layer", "[Dropout]") {
  nn::Layer::Array x = nn::Layer::Array::Ones(4, 100);
  nn::Layer::Array y(4, 100);
  nn::DropOutLayer dropout(4, 100);
  dropout.forward(y, x, true);
  nn::Layer::Array row_sum = y.rowwise().sum();
  float average = row_sum.sum() / y.rows();
  std::cout << "average dropout: " << average << std::endl;
  float tollerance = (y.cols() / 100.0f) * 10.0f;
  float min = y.cols() / 2.0f - tollerance;
  float max = y.cols() / 2.0f + tollerance;
  REQUIRE(average > min);
  REQUIRE(average < max);
}
TEST_CASE("Batch Normalization Layer", "[Batchnorm]") {
  nn::Layer::Array x = nn::Layer::Array::Random(4, 100);
  x += 100;
  nn::Layer::Array dx(4, 100);
  nn::Layer::Array dy = nn::Layer::Array::Ones(4, 100);
  nn::Layer::Array y(4, 100);
  nn::BatchnormLayer batchnorm(4, 100);
  batchnorm.forward(y, x, true);
  batchnorm.backward(dx, x, y, dy);
  batchnorm.forward(y, x, false);
}

TEST_CASE("Neural Network", "[NN]") {
  nn::NeuralNetwork net(32);
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(100, 200, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(200));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(32, 200));
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(200, 100, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(100));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(32, 100));
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(100, 10, 1e-6f));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(32, 10));
  net.addOutputLayer(std::make_unique<nn::SoftmaxLayer>(10));

  nn::Layer::Array a = nn::Layer::Array::Random(32, 100);
  nn::Layer::Array y(32, 10);
  Eigen::VectorXf labels(40, 1);
  labels << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
      3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
  float loss = net.forward(a, labels, false);
  net.backward(a, 0.01f);
  REQUIRE(y.rows() == 32);
  REQUIRE(y.cols() == 10);
  y = net.result();
  std::cout << "loss: " << loss << std::endl;
  std::cout << "array: " << y << std::endl;
}

TEST_CASE("Xor Test", "[Xor]") {
  nn::NeuralNetwork net(4);
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(2, 8, 1e-8f));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(8));
  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(8, 1, 1e-8f));
  net.addOutputLayer(std::make_unique<nn::LeastSquaresLayer>(1));

  nn::Layer::Array input = nn::Layer::Array(4, 2);
  nn::Layer::Array expected_output = nn::Layer::Array(4, 1);
  input << 0, 0, 0, 1, 1, 0, 1, 1;
  expected_output << 0, 1, 1, 0;

  // train 1000 epochs
  float loss = 0;
  for (int i = 0; i < 1000; i++) {
	loss = net.forward(input, expected_output, true);
	net.backward(input, 0.01f);
  }
  // nn::Layer::Array output = net.y().round();
  // REQUIRE(output.isApprox(expected_output));
  std::cout << "loss: " << loss << std::endl;

  nn::Layer::Array test_data1(1, 2);
  test_data1 << 0, 0;
  nn::Layer::Array test_data2(1, 2);
  test_data2 << 0, 1;
  nn::Layer::Array test_data3(1, 2);
  test_data3 << 1, 0;
  nn::Layer::Array test_data4(1, 2);
  test_data4 << 1, 1;
  nn::Layer::Array expected_result1(1, 1);
  expected_result1 << 0;
  nn::Layer::Array expected_result2(1, 1);
  expected_result2 << 1;
  nn::Layer::Array expected_result3(1, 1);
  expected_result3 << 1;
  nn::Layer::Array expected_result4(1, 1);
  expected_result4 << 0;

  net.inference(test_data1);
  nn::Layer::Array output = net.inferenceResult().round();
  REQUIRE(output.isApprox(expected_result1));
  std::cout << "0 xor 0: " << output + 0.0 << std::endl;
  net.inference(test_data2);
  output = net.inferenceResult().round();
  REQUIRE(output.isApprox(expected_result2));
  std::cout << "0 xor 1: " << output + 0.0 << std::endl;
  net.inference(test_data3);
  output = net.inferenceResult().round();
  REQUIRE(output.isApprox(expected_result3));
  std::cout << "1 xor 0: " << output + 0.0 << std::endl;
  net.inference(test_data4);
  output = net.inferenceResult().round();
  REQUIRE(output.isApprox(expected_result4));
  std::cout << "1 xor 1: " << output + 0.0 << std::endl;
}

TEST_CASE("im2row", "[im2row]") {
  nn::Layer::Array x(2, 32);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105,
      106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
      121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132;

  nn::Layer::Array dx(2, 32);
  int matrix_width;
  int matrix_height;
  int padding = nn::utils::convolution_helper::padding(3, true);
  nn::utils::convolution_helper::im2rowOutputSize(matrix_height, matrix_width,
                                                  4, 4, 2, 2, 3, padding, 1);
  nn::Layer::Array output(matrix_height, matrix_width);
  nn::Layer::Array dy = nn::Layer::Array::Ones(matrix_height, matrix_width);
  nn::utils::convolution_helper::im2row(output, x, 4, 4, 2, 2, 3, padding, 1);
  nn::utils::convolution_helper::im2rowBackward(dx, dy, 4, 4, 2, 2, 3, padding,
                                                1);
  std::cout << "im2row:\n" << x << "\n" << std::endl;
  std::cout << output << std::endl;
  std::cout << "dx:" << std::endl;
  std::cout << dx << std::endl;
}

TEST_CASE("Convolution Layer", "[conv]") {
  nn::Layer::Array x(2, 4 * 4 * 2);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105,
      106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
      121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132;
  nn::Layer::Array dx(2, 4 * 4 * 2);
  int padding = nn::utils::convolution_helper::padding(3, 1);
  nn::Layer::Array output(2, 4 * 4 * 8);
  nn::Layer::Array dy = nn::Layer::Array::Ones(2, 4 * 4 * 8);
  nn::ConvolutionLayer conv(4, 4, 2, 4, 4, 8, 3, 2, padding, 1);
  conv.forward(output, x, true);
  conv.backward(dx, x, output, dy);
  conv.update(1e-3f);
  std::cout << "conv:" << std::endl;
  std::cout << "x:" << std::endl;
  std::cout << x << std::endl;
  std::cout << "output:" << std::endl;
  std::cout << output << std::endl;
  std::cout << "dx:" << std::endl;
  std::cout << dx << std::endl;
}

TEST_CASE("maxPooling", "[maxpool]") {
  nn::Layer::Array input(2, 6 * 6 * 2);
  for (int batch = 0; batch < 2; batch++) {
	for (int y = 0; y < 6; y++) {
		for (int x = 0; x < 6; x++) {
		for (int z = 0; z < 2; z++) {
			int i = y * 12 + x * 2 + z;
		  input(batch, i) = static_cast<float>(y * 6 + x);
		}
	  }
	}
  }
  nn::Layer::Array output(2, 3 * 3 * 2);
  nn::Layer::Array dx(2, 6 * 6 * 2);
  nn::Layer::Array dy = nn::Layer::Array::Ones(2, 3 * 3 * 2);
  std::vector<uint8_t> indices(2 * 3 * 3 * 2);
  nn::utils::convolution_helper::maxPooling(output, indices, input, 6, 6, 2, 2,
                                            2);
  nn::utils::convolution_helper::maxPoolingBackward(dx, indices, dy, 6, 6, 2, 2,
                                                    2);
  std::cout << "max pooling" << std::endl;
  std::cout << "x:" << std::endl;
  std::cout << input.reshaped<Eigen::RowMajor>(12, 12) << std::endl;
  std::cout << "output:" << std::endl;
  std::cout << output.reshaped<Eigen::RowMajor>(6, 6) << std::endl;
  std::cout << "dx:" << std::endl;
  std::cout << dx.reshaped<Eigen::RowMajor>(12, 12) << std::endl;
}
TEST_CASE("averagePooling", "[averagepool]") {
  nn::Layer::Array input(2, 6 * 6 * 2);
  for (int batch = 0; batch < 2; batch++) {
	for (int y = 0; y < 6; y++) {
		for (int x = 0; x < 6; x++) {
		for (int z = 0; z < 2; z++) {
			int i = y * 12 + x * 2 + z;
		  input(batch, i) = static_cast<float>(y * 6 + x);
		}
	  }
	}
  }
  nn::Layer::Array output(2, 3 * 3 * 2);
  nn::Layer::Array dx(2, 6 * 6 * 2);
  nn::Layer::Array dy = nn::Layer::Array::Ones(2, 3 * 3 * 2);
  nn::utils::convolution_helper::averagePooling(output, input, 6, 6, 2, 2, 2);
  nn::utils::convolution_helper::averagePoolingBackward(dx, dy, 6, 6, 2, 2, 2);
  std::cout << "average pooling" << std::endl;
  std::cout << "x:" << std::endl;
  std::cout << input.reshaped<Eigen::RowMajor>(12, 12) << std::endl;
  std::cout << "output:" << std::endl;
  std::cout << output.reshaped<Eigen::RowMajor>(6, 6) << std::endl;
  std::cout << "dx:" << std::endl;
  std::cout << dx.reshaped<Eigen::RowMajor>(12, 12) << std::endl;
}
