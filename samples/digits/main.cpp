#include <layer/batchnorm_layer.h>
#include <layer/dropout_layer.h>
#include <layer/fully_connected_layer.h>
#include <layer/relu_layer.h>
#include <layer/softmax_layer.h>
#include <neural_network.h>
#include <utils/mnist/mnist_loader.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

int main(int argc, char* argv[]) {
  if (argc < 2) {
	std::cerr << "too few arguments!" << std::endl;
	std::cerr << "you have to specify the path to the dataset directory!"
	          << std::endl;
	std::exit(1);
  }
  std::string train_images_path(argv[1]);
  train_images_path += "/train-images-idx3-ubyte";
  std::string train_labels_path(argv[1]);
  train_labels_path += "/train-labels-idx1-ubyte";
  std::string test_images_path(argv[1]);
  test_images_path += "/t10k-images-idx3-ubyte";
  std::string test_labels_path(argv[1]);
  test_labels_path += "/t10k-labels-idx1-ubyte";

  nn::Layer::Array train_images;
  nn::Layer::Array train_labels;
  nn::Layer::Array test_images;
  nn::Layer::Array test_labels;
  if ((nn::utils::mnist::loadImages(train_images, train_images_path.c_str()) !=
       0) ||
      (nn::utils::mnist::loadLabels(train_labels, train_labels_path.c_str()) !=
       0) ||
      (nn::utils::mnist::loadImages(test_images, test_images_path.c_str()) !=
       0) ||
      (nn::utils::mnist::loadLabels(test_labels, test_labels_path.c_str()) !=
       0)) {
	exit(1);
  }
  int mini_batch_size = 30;
  int train_size = train_images.rows();
  int test_size = test_images.rows();
  int image_size = train_images.cols();
  int num_mini_batches = train_size / mini_batch_size;
  int num_test_mini_batches = test_size / mini_batch_size;

  if (num_mini_batches * mini_batch_size < train_size) {
	num_mini_batches++;
  }
  if (num_test_mini_batches * mini_batch_size < test_size) {
	num_test_mini_batches++;
  }

  nn::Layer::Array mini_batch_images(mini_batch_size, image_size);
  nn::Layer::Array mini_batch_label(mini_batch_size, 1);
  nn::NeuralNetwork net(mini_batch_size);
  float l2_regularization = 1e-5f;

  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(
      image_size, 1024, l2_regularization));
  net.addHiddenLayer(
      std::make_unique<nn::BatchnormLayer>(mini_batch_size, 1024));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(1024));
  //
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(1024, 256, l2_regularization));
  net.addHiddenLayer(
      std::make_unique<nn::BatchnormLayer>(mini_batch_size, 256));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(256));
  // net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(mini_batch_size,
  // 256));
  //

  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(256, 128, l2_regularization));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(128));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(mini_batch_size, 128));
  //
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(128, 32, l2_regularization));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(32));
  net.addHiddenLayer(std::make_unique<nn::DropOutLayer>(mini_batch_size, 32));
  //
  net.addHiddenLayer(
      std::make_unique<nn::FullyConnectedLayer>(32, 10, l2_regularization));
  //
  net.addOutputLayer(std::make_unique<nn::SoftmaxLayer>(10));

  std::vector<int> indices(train_size);
  for (int i = 0; i < train_size; i++) {
	indices[i] = i;
  }

  printf("Start Training...\n");
  fflush(stdout);

  std::random_device seed_generator;
  std::mt19937 mersenne_twister(seed_generator());
  std::uniform_int_distribution<uint32_t> int_distribution(0, train_size - 1);
  for (int epoch = 0; epoch < 10; epoch++) {
	// permutate
	for (int i = 0; i < indices.size(); i++) {
		int a = int_distribution(mersenne_twister);
	  int b = int_distribution(mersenne_twister);
	  std::swap(indices[a], indices[b]);
	}

	// train one epoch
	float train_loss = 0;
	for (int i = 0; i < num_mini_batches; i++) {
		for (int j = 0; j < mini_batch_size; j++) {
		int index = indices[(i * mini_batch_size + j) % train_size];
		mini_batch_images.row(j) = train_images.row(index);
		mini_batch_label.row(j) = train_labels.row(index);
	  }
	  train_loss += net.forward(mini_batch_images, mini_batch_label, true);
	  net.backward(mini_batch_images, 1e-4f);
	}
	train_loss /= num_mini_batches;
	// test
	float test_loss = 0;
	int accuracy_counter = 0;
	for (int i = 0; i < num_test_mini_batches; i++) {
		for (int j = 0; j < mini_batch_size; j++) {
		int index = (i * mini_batch_size + j) % test_size;
		mini_batch_images.row(j) = test_images.row(index);
		mini_batch_label.row(j) = test_labels.row(index);
	  }
	  float loss = net.forward(mini_batch_images, mini_batch_label, false);

	  // calculate accuracy
	  test_loss += loss;
	  const nn::Layer::ConstArrayRef& y = net.y();
	  for (int k = 0; k < mini_batch_size; k++) {
		int label = static_cast<int>(mini_batch_label(k, 0));
		int max;
		y.row(k).maxCoeff(&max);
		accuracy_counter += ((max == label) ? 1 : 0);
	  }
	}
	test_loss /= num_test_mini_batches;

	// if (epoch % 10 == 0) {
	float accuracy =
	    (static_cast<float>(accuracy_counter) / test_size) * 100.0f;
	printf("epoch:%d, accuracy:%f%%, train loss:%f, test loss:%f\n", epoch,
	       accuracy, train_loss, test_loss);
	fflush(stdout);
	//}
	if (accuracy >= 97.0f) {
		break;
	}
  }
  printf("show 10 random test images their value\n");
  std::uniform_int_distribution<uint32_t> test_distribution(0, test_size - 1);
  for (int j = 0; j < mini_batch_size; j++) {
	int index = test_distribution(mersenne_twister);
	mini_batch_images.row(j) = test_images.row(index);
	mini_batch_label.row(j) = test_labels.row(index);
  }
  float loss = net.forward(mini_batch_images, mini_batch_label, false);

  for (int i = 0; i < 10; i++) {
	int max;
	net.y().row(i).maxCoeff(&max);
	int ground_truth = mini_batch_label(i, 0);
	printf("result: %d, ground truth: %d\n", max, ground_truth);
	fflush(stdout);
  }

  return 0;
}
