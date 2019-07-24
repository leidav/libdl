#include "convolution_helper.h"
#include <omp.h>
#include <array>
#include <limits>

namespace nn {
namespace utils {
namespace convolution_helper {
int imageoutputWidth(int input_width, int input_height, int kernel_size,
                     int padding, int stride) {
  return (input_width - kernel_size + padding * 2) / stride + 1;
}
int imageoutputHeight(int input_width, int input_height, int kernel_size,
                      int padding, int stride) {
  return (input_height - kernel_size + padding * 2) / stride + 1;
}

void imageoutputSize(int &output_width, int &output_height, int input_width,
                     int input_height, int kernel_size, int padding,
                     int stride) {
  output_width =
      imageoutputWidth(input_width, input_height, kernel_size, padding, stride);
  output_height = imageoutputHeight(input_width, input_height, kernel_size,
                                    padding, stride);
}

void im2row(Layer::ArrayRef out, const Layer::ConstArrayRef &in, int width,
            int height, int depth, int batch_size, int kernel_size, int padding,
            int stride) {
  auto index = [width](int x, int y) -> int { return y * width + x; };
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, width, height, kernel_size,
                  padding, stride);

  int image_size = width * height;

  Eigen::Map<const Layer::Array> reshaped_in(in.data(), image_size * batch_size,
                                             depth);
  // const Layer::ConstArrayRef &reshaped_in =
  //   in.reshaped<Eigen::RowMajor>(image_size * batch_size, depth);

  int rows = out.rows() / batch_size;
#pragma omp parallel for
  for (int batch = 0; batch < batch_size; batch++) {
	int image_offset = batch * image_size;
	int output_x = 0;
	int x_offset = -padding;
	int y_offset = -padding;
	for (int row = 0; row < rows; row++) {
		int col = 0;
	  for (int y = y_offset; y < (y_offset + kernel_size); y++) {
		for (int x = x_offset; x < (x_offset + kernel_size); x++) {
			if ((x < 0) || (x >= width) || (y < 0) || (y >= height)) {
			out.block(batch * rows + row, col, 1, depth).setZero();
		  } else {
			int i = image_offset + index(x, y);
			out.block(batch * rows + row, col, 1, depth) = reshaped_in.row(i);
		  }
		  col += depth;
		}
	  }
	  if (output_x == output_width - 1) {
		output_x = 0;
		x_offset = 0;
		y_offset += stride;
	  } else {
		output_x++;
		x_offset += stride;
	  }
	}
  }
}
int im2rowOutputRows(int image_width, int image_height, int image_depth,
                     int batch_size, int kernel_size, int padding, int stride) {
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, image_width, image_height,
                  kernel_size, padding, stride);

  return output_width * output_height * batch_size;
}

int im2rowOutputCols(int image_width, int image_height, int image_depth,
                     int batch_size, int kernel_size, int padding, int stride) {
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, image_width, image_height,
                  kernel_size, padding, stride);

  return kernel_size * kernel_size * image_depth;
}

void im2rowOutputSize(int &output_rows, int &output_cols, int image_width,
                      int image_height, int image_depth, int batch_size,
                      int kernel_size, int padding, int stride) {
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, image_width, image_height,
                  kernel_size, padding, stride);

  output_cols = kernel_size * kernel_size * image_depth;
  output_rows = output_width * output_height * batch_size;
}

void filterSize(int &rows, int &cols, int input_depth, int kernel_size,
                int num_kernels) {
  rows = filterRows(input_depth, kernel_size, num_kernels);
  cols = filterCols(input_depth, kernel_size, num_kernels);
}

int padding(int kernel_size, bool keep_size) {
  if (keep_size) {
	return (kernel_size - 1) / 2;
  }
  return 0;
}

int filterCols(int input_depth, int kernel_size, int num_kernels) {
  return num_kernels;
}

int filterRows(int input_depth, int kernel_size, int num_kernels) {
  return input_depth * kernel_size * kernel_size;
}

void im2rowBackward(Layer::ArrayRef dx, const Layer::ConstArrayRef &dy,
                    int width, int height, int depth, int batch_size,
                    int kernel_size, int padding, int stride) {
  dx.setZero();

  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, width, height, kernel_size,
                  padding, stride);

  auto index = [width](int x, int y) -> int { return y * width + x; };

  int image_size = width * height;

  Eigen::Map<Layer::Array> reshaped_dx(dx.data(), image_size * batch_size,
                                       depth);

  int rows = dy.rows() / batch_size;

#pragma omp parallel for
  for (int batch = 0; batch < batch_size; batch++) {
	int image_offset = batch * image_size;
	int output_x = 0;
	int x_offset = -padding;
	int y_offset = -padding;
	for (int row = 0; row < rows; row++) {
		int col = 0;
	  for (int y = y_offset; y < (y_offset + kernel_size); y++) {
		for (int x = x_offset; x < (x_offset + kernel_size); x++) {
			if ((x >= 0) && (x < width) && (y >= 0) && (y < height)) {
			int i = image_offset + index(x, y);
			reshaped_dx.row(i) += dy.block(batch * rows + row, col, 1, depth);
		  }
		  col += depth;
		}
	  }
	  if (output_x == output_width - 1) {
		output_x = 0;
		x_offset = 0;
		y_offset += stride;
	  } else {
		output_x++;
		x_offset += stride;
	  }
	}
  }
}

bool isValid(int width, int height, int kernel_size, int padding, int stride) {}

void maxPooling(Layer::ArrayRef out, std::vector<uint8_t> &indices,
                const Layer::ConstArrayRef &in, int width, int height,
                int depth, int batch_size, int kernel_size) {
  int stride = kernel_size;
  auto input_index = [width, depth](int x, int y, int z) -> int {
	return y * width * depth + x * depth + z;
  };
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, width, height, kernel_size, 0,
                  stride);
  auto output_index = [output_width, depth](int x, int y, int z) -> int {
	return y * output_width * depth + x * depth + z;
  };

  int output_size = output_width * output_height * depth;
  out.setConstant(std::numeric_limits<float>::min());

#pragma omp parallel for
  for (int batch = 0; batch < batch_size; batch++) {
	for (int output_y = 0; output_y < output_height; output_y++) {
		int y_offset = output_y * stride;
	  for (int output_x = 0; output_x < output_width; output_x++) {
		int x_offset = output_x * stride;
		for (int y = y_offset; y < (y_offset + kernel_size); y++) {
			for (int x = x_offset; x < (x_offset + kernel_size); x++) {
			int ii = input_index(x, y, 0);
			int oi = output_index(output_x, output_y, 0);
			for (int z = 0; z < depth; z++) {
				float val = in(batch, ii);
			  float max = out(batch, oi);
			  if (val > max) {
				out(batch, oi) = val;
				// std::get<0>(indices[batch * output_size +
				// oi]) = x - x_offset;
				// std::get<1>(indices[batch * output_size +
				// oi]) = y
				// - y_offset;
				uint8_t index = ((x - x_offset) << 4) | (0xF & (y - y_offset));
				indices[batch * output_size + oi] = index;
			  }
			  ii++;
			  oi++;
			}
		  }
		}
	  }
	}
  }
}

void maxPoolingBackward(Layer::ArrayRef dx, std::vector<uint8_t> &indices,
                        const Layer::ConstArrayRef &dy, int width, int height,
                        int depth, int batch_size, int kernel_size) {
  int stride = kernel_size;
  auto input_index = [width, depth](int x, int y, int z) -> int {
	return y * width * depth + x * depth + z;
  };
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, width, height, kernel_size, 0,
                  stride);
  auto output_index = [output_width, depth](int x, int y, int z) -> int {
	return y * output_width * depth + x * depth + z;
  };
  int output_size = output_width * output_height * depth;
  dx.setZero();
#pragma omp parallel for
  for (int batch = 0; batch < batch_size; batch++) {
	for (int output_y = 0; output_y < output_height; output_y++) {
		int y_offset = output_y * stride;
	  for (int output_x = 0; output_x < output_width; output_x++) {
		int x_offset = output_x * stride;
		int dyi = output_index(output_x, output_y, 0);
		for (int z = 0; z < depth; z++) {
			uint8_t index = indices[batch * output_size + dyi];
		  int x = static_cast<int>(index >> 4) + x_offset;
		  int y = static_cast<int>(index & 0xF) + y_offset;
		  // int x = std::get<0>(indices[batch * output_size + dyi]) +
		  // x_offset; int y = std::get<1>(indices[batch * output_size
		  // + dyi]) + y_offset;
		  dx(batch, input_index(x, y, z)) = dy(batch, dyi);
		  dyi++;
		}
	  }
	}
  }
}

void averagePooling(Layer::ArrayRef out, const Layer::ConstArrayRef &in,
                    int width, int height, int depth, int batch_size,
                    int kernel_size) {
  int stride = kernel_size;
  auto input_index = [width, depth](int x, int y, int z) -> int {
	return y * width * depth + x * depth + z;
  };
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, width, height, kernel_size, 0,
                  stride);
  auto output_index = [output_width, depth](int x, int y, int z) -> int {
	return y * output_width * depth + x * depth + z;
  };

  out.setZero();
  float w = 1.0f / (kernel_size * kernel_size);

#pragma omp parallel for
  for (int batch = 0; batch < batch_size; batch++) {
	for (int output_y = 0; output_y < output_height; output_y++) {
		int y_offset = output_y * stride;
	  for (int output_x = 0; output_x < output_width; output_x++) {
		int x_offset = output_x * stride;
		for (int y = y_offset; y < (y_offset + kernel_size); y++) {
			for (int x = x_offset; x < (x_offset + kernel_size); x++) {
			int ii = input_index(x, y, 0);
			int oi = output_index(output_x, output_y, 0);
			for (int z = 0; z < depth; z++) {
				out(batch, oi) += in(batch, ii) * w;
			  ii++;
			  oi++;
			}
		  }
		}
	  }
	}
  }
}

void averagePoolingBackward(Layer::ArrayRef dx, const Layer::ConstArrayRef &dy,
                            int width, int height, int depth, int batch_size,
                            int kernel_size) {
  int stride = kernel_size;
  auto input_index = [width, depth](int x, int y, int z) -> int {
	return y * width * depth + x * depth + z;
  };
  int output_width;
  int output_height;
  imageoutputSize(output_width, output_height, width, height, kernel_size, 0,
                  stride);
  auto output_index = [output_width, depth](int x, int y, int z) -> int {
	return y * output_width * depth + x * depth + z;
  };

  float w = 1.0f / (kernel_size * kernel_size);

#pragma omp parallel for
  for (int batch = 0; batch < batch_size; batch++) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
		int ii = input_index(x, y, 0);
		int oi = output_index(x / kernel_size, y / kernel_size, 0);
		for (int z = 0; z < depth; z++) {
			dx(batch, ii) = dy(batch, oi) * w;
		  ii++;
		  oi++;
		}
	  }
	}
  }
}

};  // namespace convolution_helper
};  // namespace utils
};  // namespace nn
