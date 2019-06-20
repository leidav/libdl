#include <layer/average_pooling_layer.h>
#include <layer/batchnorm_layer.h>
#include <layer/convolution_layer.h>
#include <layer/dropout_layer.h>
#include <layer/fully_connected_layer.h>
#include <layer/max_pooling_layer.h>
#include <layer/relu_layer.h>
#include <layer/softmax_layer.h>
#include <neural_network.h>
#include <utils/convolution_helper.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

static uint32_t swapEndian(uint32_t integer) {
  uint32_t tmp = ((integer >> 24) & 0x000000FF);
  tmp |= ((integer >> 8) & 0x0000FF00);
  tmp |= ((integer << 8) & 0x00FF0000);
  tmp |= ((integer << 24) & 0xFF000000);
  return tmp;
}
static int loadImages(nn::Layer::Array& images, const char* path) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
	return -1;
  }
  uint32_t magic_number = 0;
  uint32_t image_count;
  uint32_t width;
  uint32_t height;
  if ((std::fread(&magic_number, sizeof(magic_number), 1, file) != 1) ||
      (std::fread(&image_count, sizeof(image_count), 1, file) != 1) ||
      (std::fread(&width, sizeof(width), 1, file) != 1) ||
      (std::fread(&height, sizeof(height), 1, file) != 1)) {
	fclose(file);
	return -1;
  }
  magic_number = swapEndian(magic_number);
  image_count = swapEndian(image_count);
  width = swapEndian(width);
  height = swapEndian(height);

  if (magic_number != 0x803) {
	fclose(file);
	return -1;
  }

  size_t image_size = width * height;
  size_t buffer_size = image_count * image_size;
  uint8_t* buffer = new uint8_t[buffer_size];
  if (fread(buffer, 1, buffer_size, file) != buffer_size) {
	fclose(file);
	return -1;
  }
  images = nn::Layer::Array(image_count, image_size);

  for (int i = 0; i < image_count; i++) {
	for (int j = 0; j < image_size; j++) {
		images(i, j) = static_cast<float>(buffer[i * image_size + j]) / 255.0f;
	}
  }
  delete[] buffer;
  fclose(file);

  return 0;
}

static int loadLabels(nn::Layer::Array& labels, const char* path) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr) {
	return -1;
  }
  uint32_t magic_number;
  uint32_t label_count;
  if ((std::fread(&magic_number, sizeof(magic_number), 1, file) != 1) ||
      (std::fread(&label_count, sizeof(label_count), 1, file) != 1)) {
	fclose(file);
	return -1;
  }
  magic_number = swapEndian(magic_number);
  label_count = swapEndian(label_count);

  if (magic_number != 0x801) {
	fclose(file);
	return -1;
  }
  uint8_t* buffer = new uint8_t[label_count];
  if (fread(buffer, 1, label_count, file) != label_count) {
	fclose(file);
	return -1;
  }
  labels = nn::Layer::Array(label_count, 1);

  for (int i = 0; i < label_count; i++) {
	labels(i, 0) = static_cast<float>(buffer[i]);
  }
  delete[] buffer;
  fclose(file);

  return 0;
}

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
  if ((loadImages(train_images, train_images_path.c_str()) != 0) ||
      (loadLabels(train_labels, train_labels_path.c_str()) != 0) ||
      (loadImages(test_images, test_images_path.c_str()) != 0) ||
      (loadLabels(test_labels, test_labels_path.c_str()) != 0)) {
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

  int input_width = 28;
  int input_height = 28;
  int output_width;
  int output_height;
  nn::convolution_helper::imageoutputSize(output_width, output_height,
                                          input_width, input_height, 5, 0, 1);
  net.addHiddenLayer(std::make_unique<nn::ConvolutionLayer>(
      input_width, input_height, 1, output_width, output_height, 6, 5,
      mini_batch_size, 0, 1, l2_regularization));
  input_width = output_width;
  input_height = output_height;
  net.addHiddenLayer(std::make_unique<nn::BatchnormLayer>(
      mini_batch_size, input_width * input_height * 6));
  net.addHiddenLayer(
      std::make_unique<nn::ReLULayer>(input_width * input_height * 6));

  nn::convolution_helper::imageoutputSize(output_width, output_height,
                                          input_width, input_height, 2, 0, 2);
  net.addHiddenLayer(std::make_unique<nn::MaxPoolingLayer>(
      input_width, input_height, 6, output_width, output_height, 2,
      mini_batch_size));

  input_width = output_width;
  input_height = output_height;

  nn::convolution_helper::imageoutputSize(output_width, output_height,
                                          input_width, input_height, 5, 0, 1);
  net.addHiddenLayer(std::make_unique<nn::ConvolutionLayer>(
      input_width, input_height, 6, output_width, output_height, 16, 5,
      mini_batch_size, 0, 1, l2_regularization));
  input_width = output_width;
  input_height = output_height;
  net.addHiddenLayer(std::make_unique<nn::BatchnormLayer>(
      mini_batch_size, input_width * input_height * 16));
  net.addHiddenLayer(
      std::make_unique<nn::ReLULayer>(input_width * input_height * 16));

  nn::convolution_helper::imageoutputSize(output_width, output_height,
                                          input_width, input_height, 2, 0, 2);
  net.addHiddenLayer(std::make_unique<nn::MaxPoolingLayer>(
      input_width, input_height, 16, output_width, output_height, 2,
      mini_batch_size));
  input_width = output_width;
  input_height = output_height;

  net.addHiddenLayer(std::make_unique<nn::FullyConnectedLayer>(
      input_width * input_height * 16, 128, l2_regularization));
  net.addHiddenLayer(std::make_unique<nn::ReLULayer>(128));
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

  std::random_device seed_generator;
  std::mt19937 mersenne_twister(seed_generator());
  std::uniform_int_distribution<uint32_t> int_distribution(0, train_size - 1);
  for (int epoch = 0; epoch < 10000; epoch++) {
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
	  if (loss >= std::numeric_limits<float>::infinity()) {
		printf("infinity in iteration i=%d\n", i);
	  }
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
  }

  return 0;
}
