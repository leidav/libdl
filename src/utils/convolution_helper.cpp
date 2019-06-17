#include "convolution_helper.h"
namespace nn {
namespace convolution_helper {
static void imageoutputSize(int &output_width, int &output_height,
                            int input_width, int input_height, int kernel_size,
                            int padding, int stride) {
  output_width = (input_width - kernel_size + padding * 2) / stride + 1;
  output_height = (input_height - kernel_size + padding * 2) / stride + 1;
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

  const Layer::ConstArrayRef &reshaped_in =
      in.reshaped<Eigen::RowMajor>(image_size * batch_size, depth);

  int rows = out.rows() / batch_size;
  int image_offset = 0;
  for (int batch = 0; batch < batch_size; batch++) {
	int output_x = 0;
	int x_offset = -padding;
	int y_offset = -padding;
	for (int row = 0; row < rows; row++) {
		int col = 0;
	  for (int y = y_offset; y < (y_offset + kernel_size); y++) {
		for (int x = x_offset; x < (x_offset + kernel_size); x++) {
			int i = image_offset + index(x, y);
		  if ((x < 0) || (x >= output_width) || (y < 0) ||
		      (y >= output_height)) {
			/*for (int z = 0; z < depth; z++) {
* rows + row, col) = 0.0f; col++;
}*/
			out.block(batch * rows + row, col, 1, depth).setZero();
		  } else {
			/*for (int z = 0; z < depth; z++) {
			* rows + row, col) = reshaped_in(i, z); col++;
			}*/
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
	image_offset += image_size;
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

void im2rowFilterSize(int &rows, int &cols, int input_depth, int kernel_size,
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

void im2row_backward(Layer::ArrayRef dx, const Layer::ConstArrayRef &dy,
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
  int image_offset = 0;

  for (int batch = 0; batch < batch_size; batch++) {
	int output_x = 0;
	int x_offset = -padding;
	int y_offset = -padding;
	for (int row = 0; row < rows; row++) {
		int col = 0;
	  for (int y = y_offset; y < (y_offset + kernel_size); y++) {
		for (int x = x_offset; x < (x_offset + kernel_size); x++) {
			int i = image_offset + index(x, y);
		  if ((x >= 0) && (x < output_width) && (y >= 0) &&
		      (y < output_height)) {
			// out.block(batch * rows + row, col, 1, depth) =
			// reshaped_in.row(i);
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
	image_offset += image_size;
  }
}

};  // namespace convolution_helper
};  // namespace nn
