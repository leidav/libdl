#include "mnist_loader.h"

#include <cstdint>
#include <cstdio>

static uint32_t swapEndian(uint32_t integer) {
  uint32_t tmp = ((integer >> 24) & 0x000000FF);
  tmp |= ((integer >> 8) & 0x0000FF00);
  tmp |= ((integer << 8) & 0x00FF0000);
  tmp |= ((integer << 24) & 0xFF000000);
  return tmp;
}

namespace nn {
namespace utils {
namespace mnist {

int loadImages(nn::Layer::Array& images, const char* path) {
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

int loadLabels(nn::Layer::Array& labels, const char* path) {
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
};  // namespace mnist
};  // namespace utils
};  // namespace nn
